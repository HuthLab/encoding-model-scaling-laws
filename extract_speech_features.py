#!/usr/bin/env python3

"""
Feature extraction for ASR models supported by Hugging Face.
"""

import argparse
import collections
import copy # for freezing specific parts of networks during randomization
import itertools
import json
import os
import operator
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import cottoncandy as cc
import ipdb
import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from transformers import AutoModel, AutoModelForPreTraining, PreTrainedModel,\
                         AutoFeatureExtractor, WhisperModel

try:
    import database_utils
    IS_STIMULIDB_AVAILABLE = True
except:
    IS_STIMULIDB_AVAILABLE = False

# Resample to this sample rate. 16kHz is used by most models.
# TODO: programatically verify the input sample rate of each model?
TARGET_SAMPLE_RATE = 16000

def extract_speech_features(model: PreTrainedModel, model_config: dict, wav: torch.Tensor,
                            chunksz_sec: float, contextsz_sec: float,
                            num_sel_frames = 1, frame_skip = 5, sel_layers: Optional[List[int]]=None,
                            batchsz: int = 1,
                            return_numpy: bool = True, move_to_cpu: bool = True,
                            disable_tqdm: bool = False, feature_extractor=None,
                            sampling_rate: int = TARGET_SAMPLE_RATE, require_full_context: bool = False,
                            stereo: bool = False):
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"
    if return_numpy: assert move_to_cpu, "'move_to_cpu' must be true if returning numpy arrays"

    # Whisper needs special handling
    is_whisper_model = isinstance(model, WhisperModel)

    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

    # Add all snippets that are exactly the length of the requested input
    # (`Tensor.unfold` is basically a sliding window).
    if wav.shape[0] >= chunksz_samples+contextsz_samples:
        # `unfold` fails if `wav.shape[0]` is less than the window size.
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, chunksz_samples+contextsz_samples, chunksz_samples)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting `require_full_context=True`")

    # 2-D array where `[i,0]` and `[i,1]` are the start and end, respectively,
    # of snippet `i` in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-(contextsz_samples+chunksz_samples)),
                                 snippet_ends], dim=1)

    # Remove snippets that are not long enough. (Seems easier to filter
    # after generating the snippet bounds than handling it above in each case)
    # TODO: is there any way to programatically check this in HuggingFace?
    # doesn't seem so (unlike s3prl).
    if 'min_input_length' in model_config:
        # this is stored originally in **samples**!!!
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model.config:
        min_length_samples = model.config['win_ms'] / 1000. * TARGET_SAMPLE_RATE

    snippet_times = snippet_times[(snippet_times[:,1] - snippet_times[:,0]) >= min_length_samples]
    snippet_times_sec = snippet_times / sampling_rate # snippet_times, but in sec.

    module_features = collections.defaultdict(list)
    out_features = [] # the final output of the model
    times = [] # times are shared across all layers

    #assert (frames_per_chunk % frame_skip) == 0, "These must be divisible"
    frame_len_sec = model_config['stride'] / TARGET_SAMPLE_RATE # length of an output frame (sec.)

    snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
    if require_full_context:
        assert all(snippet_length_samples == snippet_length_samples[0]), "uneven snippet lengths!"
        snippet_length_samples = snippet_length_samples[0]
        assert snippet_length_samples.ndim == 0

    # Set up the iterator over batches of snippets
    if require_full_context:
        # This case is simpler, so handle it explicitly
        snippet_batches = snippet_times.T.split(batchsz, dim=1)
    else:
        # First, batch the snippets that are of different lengths.
        snippet_batches = snippet_times.tensor_split(torch.where(snippet_length_samples.diff() != 0)[0]+1, dim=0)
        # Then, split any batches that are too big to fit into the given
        # batch size.
        snippet_iter = []
        for batch in snippet_batches:
            # split, *then* transpose
            if batch.shape[0] > batchsz:
                snippet_iter += batch.T.split(batchsz,dim=1)
            else:
                snippet_iter += [batch.T]
        snippet_batches = snippet_iter

    snippet_iter = snippet_batches
    if not disable_tqdm:
        snippet_iter = tqdm(snippet_iter, desc='snippet batches', leave=False)
    snippet_iter = enumerate(snippet_iter)


    # Iterate with a sliding window. stride = chunk_sz
    for batch_idx, (snippet_starts, snippet_ends) in snippet_iter:
        if ((snippet_ends - snippet_starts) < (contextsz_samples + chunksz_samples)).any() and require_full_context:
            raise ValueError("This shouldn't happen with require_full_context")

        # If we don't have enough samples, skip this chunk.
        if (snippet_ends - snippet_starts < min_length_samples).any():
            print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet')
            assert False

        # Construct the input waveforms for the batch
        batched_wav_in_list = []
        for batch_snippet_idx, (snippet_start, snippet_end) in enumerate(zip(snippet_starts, snippet_ends)):
            # Stacking might be inefficient, so populate a pre-allocated array.
            #batched_wav_in[batch_snippet_idx, :] = wav[snippet_start:snippet_end]
            # But stacking makes variable batch size easier!
            batched_wav_in_list.append(wav[snippet_start:snippet_end])
        batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

        # The final batch may be incomplete if batchsz doesn't evenly divide
        # the number of snippets.
        if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != batchsz):
            batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        # Take the last 1 or 2 activations, and time-wise put it at the
        # end of chunk.
        output_inds = np.array([-1 - frame_skip*i for i in reversed(range(num_sel_frames))])

        # Use a pre-processor if given (e.g. to normalize the waveform), and
        # then feed into the model.
        if feature_extractor is not None:
            # This step seems to be NOT differentiable, since the feature
            # extractor first converts the Tensor to a numpy array, then back
            # into a Tensor.
            # If you want to backprop through the stimulus, you might have to
            # re-implement the feature extraction in PyTorch (in particular, the
            # normalization)

            if stereo: raise NotImplementedError("Support handling multi-channel audio with feature extractor")
            # It looks like most feature extractors (e.g.
            # Wav2Vec2FeatureExtractor) accept mono audio (i.e. 1-dimensional),
            # but it's unclear if they support stereo as well.

            feature_extractor_kwargs = {}
            if is_whisper_model:
                # Because Whisper auto-pads all inputs to 30 sec., we'll use
                # the attention mask to figure out when the "last" relevant
                # input was.
                features_key = 'input_features'
                feature_extractor_kwargs['return_attention_mask'] = True
            else:
                features_key = 'input_values'

            preprocessed_snippets = feature_extractor(list(batched_wav_in.cpu().numpy()),
                                                      return_tensors='pt',
                                                      sampling_rate=sampling_rate,
                                                      **feature_extractor_kwargs)
            if is_whisper_model:
                chunk_features = model.encoder(preprocessed_snippets[features_key].to(model.device))

                # Now we need to figure out which output index to use, since 2
                # conv layers downsample the inputs before passing them into
                # the encoder's Transformer layers. We can redo the encoder's
                # 1-D conv's on the attention mask to find the final output that
                # was influenced by the snippet.
                contributing_outs = preprocessed_snippets.attention_mask # 1 if part of waveform, 0 otherwise. shape: (batchsz, 3000)
                # Taking [0] works because all snippets have the same length.
                # Add the dimension back for `conv1d` to work
                # TODO: assert that all clips are the same length?
                contributing_outs = contributing_outs[0].unsqueeze(0)

                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv1.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv1.stride,
                                                               padding=model.encoder.conv1.padding,
                                                               dilation=model.encoder.conv1.dilation,
                                                               groups=model.encoder.conv1.groups)
                # shape: (batchsz, 1500)
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv2.stride,
                                                               padding=model.encoder.conv2.padding,
                                                               dilation=model.encoder.conv2.dilation,
                                                               groups=model.encoder.conv1.groups)

                final_output = contributing_outs[0].nonzero().squeeze(-1).max()
            else:
                # sampling rates must match if not using a pre-processor
                assert sampling_rate == TARGET_SAMPLE_RATE, f"sampling rate mismatch! {sampling_rate} != {TARGET_SAMPLE_RATE}"

                chunk_features = model(preprocessed_snippets[features_key].to(model.device))
        else:
            chunk_features = model(batched_wav_in)

        # Make sure we have enough outputs
        if(chunk_features['last_hidden_state'].shape[1] < (num_sel_frames-1) * frame_skip - 1):
            print("Skipping:", snippet_idx, "only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (num_sel_frames-1) * frame_skip - 1, "were needed.")
            continue

        assert len(output_inds) == 1, "Only one output per evaluation is "\
            "supported for Hugging Face (because they don't provide the downsampling rate)"

        if is_whisper_model:
            output_inds = [final_output]

        for out_idx, output_offset in enumerate(output_inds):
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            output_representation = chunk_features['last_hidden_state'][:, output_offset, :] # shape: (batchsz, hidden_size)
            if move_to_cpu: output_representation = output_representation.cpu()
            if return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            # Collect features from individual layers
            # NOTE: outs['hidden_states'] might have an extra element at
            # the beginning for the feature extractor.
            # e.g. 25 "layers" --> CNN output + 24 transformer layers' output
            for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
                # Only save layers that the user wants (if specified)
                if sel_layers:
                    if layer_idx not in sel_layers: continue

                layer_representation = layer_activations[:, output_offset, :] # shape: (batchsz, hidden_size)
                if move_to_cpu: layer_representation = layer_representation.cpu()
                if return_numpy: layer_representation = layer_representation.numpy() # TODO: convert to numpy at the end

                if is_whisper_model:
                    # Leave the option open for using decoder layers in the
                    # future
                    module_name = f"encoder.{layer_idx}"
                else:
                    module_name = f"layer.{layer_idx}"

                module_features[module_name].append(layer_representation)

    out_features = np.concatenate(out_features, axis=0) if return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
    module_features = {name: (np.concatenate(features, axis=0) if return_numpy else torch.cat(features, dim=0))\
                       for name, features in module_features.items()}

    assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
        "Missing timesteps in the module activations!! (possible PyTorch bug)"
    times = torch.cat(times, dim=0) / TARGET_SAMPLE_RATE # convert samples --> seconds. shape: (timesteps,)
    if return_numpy: times = times.numpy()

    del chunk_features # possible memory leak. remove if unneeded
    return {'final_outputs': out_features, 'times': times,
            'module_features': module_features}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimulus_dir', type=Path,
                        default='./processed_stimuli/',
                        help="Directory with preprocessed stimuli wav's.")
    parser.add_argument('--bucket', type=str,
                        help="Bucket to save extracted features to. If blank, save to local filesystem.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--use_featext', action='store_true')
    parser.add_argument('--batchsz', type=int, default=1,
                        help='Number of audio clips to evaluate at once. (Only uses one GPU.)')
    parser.add_argument('--chunksz', type=float, default=100,
                        help="Divide the stimulus waveform into chunks of this many *milliseconds*.")
    parser.add_argument('--contextsz', type=float, default=8000,
                        help="Use these many milliseconds as context for each chunk.")
    parser.add_argument('--layers', nargs='+', type=int, help="Only save the "
                        "features from these layers. Usually doesn't speed up execution "
                        "time, but may speed up upload time and reduce total disk usage. "
                        "NOTE: only works with numbered layers (currently).")
    parser.add_argument('--full_context', action='store_true',
                        help="Only extract the representation for a stimulus if it is as long as the feature extractor's specified context (context_sz)")
    parser.add_argument('--resample', action='store_true',
                        help='Resample the stimuli to the necessary sample rate '
                        'and convert stereo to mono if needed. If this flag is '
                        'not supplied, an assertion will fail if either '
                        'condition is not met.')
    parser.add_argument('--stride', type=float,
                        help='Extract features every <n> seconds. If using --custom_stimuli, consider changing this argument. Don\'t use this for extracting story features to train encoding models (use --chunksz instead). 0.5 is a good value.')
    parser.add_argument('--pad_silence', action='store_true',
                        help='Pad short clips (less than context_sz+chunk_sz) with silence at the beginning')

    # Arguments for choosing stories
    stimulus_sel_args = parser.add_argument_group('stimulus_sel', 'Stimulus selection')
    stimulus_sel_args.add_argument('--sessions', nargs='+', type=str,
                                   help="Only process stories presented in these sessions."
                                   "Can be used in conjuction with --subject to take an intersection.")
    stimulus_sel_args.add_argument('--stories', '--stimuli', nargs='+', type=str,
                                   help="Only process the given stories."
                                   "Overrides --sessions and --subjects.")
    stimulus_sel_args.add_argument('--recursive', action='store_true',
                                   help='Recursively find .wav and .flac in the stimulus_dir.')
    stimulus_sel_args.add_argument('--custom_stimuli', type=str,
                                    help='Use custom (non-story) stimuli, stored in '
                                    '"{stimulus_dir}/{custom_stimuli}". If this flag '
                                    'is not set, use story stimuli.')
    stimulus_sel_args.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing features (default behavior is to skip)')


    args = parser.parse_args()

    if (args.bucket is not None) and (args.bucket != ''):
        cci_features = cc.get_interface(args.bucket, verbose=False)
        print("Saving features to bucket", cci_features.bucket_name)
    else:
        cci_features = None
        print('Saving features to local filesystem.')
        print('NOTE: You can use ./upload_features_to_corral.sh to upload them later if you wish')

    model_name = args.model
    with open('speech_model_configs.json', 'r') as f:
        model_config = json.load(f)[model_name]
        model_hf_path = model_config['huggingface_hub']
    print('Loading model', model_name, 'from the Hugging Face Hub...')
    model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True).cuda()
    feature_extractor = None
    if args.use_featext:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)

    ## Stimulus selection
    # Using CLI arguments, find stimuli and their locations.
    stories = set()

    if args.stories is not None:
        stories.update(args.stories)

    if args.sessions is not None:
        assert IS_STIMULIDB_AVAILABLE, "database_utils is unavailable but is needed to access Stimuli DB"
        cci_stim = cc.get_interface('stimulidb', verbose=False)
        sess_to_story = cci_stim.download_json('sess_to_story') # IMO this should be added to database_utils

        for session in args.sessions:
            train_stories, test_story = sess_to_story[session]
            stories.add(test_story)
            for story in train_stories:
                stories.add(story)

    stimulus_dir = args.stimulus_dir
    assert stimulus_dir.exists(), f"Stimulus dir {str(stimulus_dir)} does not exist"
    assert stimulus_dir.is_dir(), f"Stimulus dir {str(stimulus_dir)} is not a directory"

    stimulus_paths: Dict[str, Path] = {} # map of stimulus name --> file path. We also use this as the list of stimuli

    if args.custom_stimuli: # optionally use non-story stimuli
        custom_stimuli_dir = stimulus_dir / args.custom_stimuli
        assert custom_stimuli_dir.exists(), f"dir {str(custom_stimuli_dir)} does not exist"
        stimulus_dir = custom_stimuli_dir

    # We haven't selected any stories yet, so just select all stories in the
    # stimulus directory.
    if len(stories) == 0:
        # Look for all files ending in '.flac' and '.wav'. If there are two
        # files with the same basename (i.e. without the suffix), then prefer
        # the FLAC file.
        if args.recursive:
            stimulus_glob_wav_iter = stimulus_dir.rglob('*.wav')
            stimulus_glob_flac_iter = stimulus_dir.rglob('*.flac')
        else:
            stimulus_glob_wav_iter = stimulus_dir.glob('*.wav')
            stimulus_glob_flac_iter = stimulus_dir.glob('*.flac')

        for stimulus_path in itertools.chain(stimulus_glob_wav_iter, stimulus_glob_flac_iter):
            # Use 'relative_to' to preserve directory structure when using
            # --recursive
            stimulus_name = str(stimulus_path.relative_to(stimulus_dir).with_suffix(''))
            # If stimulus already exists, overwrite the path with the
            # most recent extension
            stimulus_paths[stimulus_name] = stimulus_path
    else:
        for story in stories:
            # Find the associated sound file for each stimulus.
            # First extension found is preferred.
            for ext in ['flac', 'wav']:
                stimulus_path = stimulus_dir / f"{story}.{ext}"
                if stimulus_path.exists() and stimulus_path.is_file():
                    stimulus_paths[story] = stimulus_path
                    break

        missing_stories = set(stories).difference(set(stimulus_paths.keys()))
        if len(missing_stories) > 0:
            raise RuntimeError(f"missing stimuli for stories: " + ' '.join(missing_stories))

    assert len(stimulus_paths) > 0, "no stimuli to process!"

    # Make sure that all preprocessed stimuli exist and are readable.
    for stimulus_name, stimulus_local_path in stimulus_paths.items():
        wav, sample_rate = torchaudio.load(stimulus_local_path)
        if not args.resample:
            assert wav.shape[0] == 1, f"stimulus '{stimulus_local_path}' is not mono-channel"

    # chunk size in seconds and samples, respectively
    chunksz_sec = args.chunksz / 1000.

    # context size in terms of chunks
    assert (args.contextsz % args.chunksz) == 0, "These must be divisible"
    contextsz_sec = args.contextsz / 1000.

    model_save_path = f"features_cnk{chunksz_sec:0.1f}_ctx{contextsz_sec:0.1f}/{model_name}"
    if args.stride:
        # If using a custom stride length (e.g. for snippets), store in a
        # separate directory.
        model_save_path = os.path.join(model_save_path, f"stride_{args.stride}")
    if args.custom_stimuli:
        # Save custom (non-story) stimuli in their own subdirectory
        model_save_path = os.path.join(model_save_path, 'custom_stimuli', args.custom_stimuli)
    print('Saving features to:', model_save_path)

    ## Feature extraction loop
    # Go through each stimulus and save resulting features
    torch.set_grad_enabled(False) # VERY important! (for memory)
    model.eval()
    # Sort stimuli alphabetically. Allows us, in theory, to resume partial/failed jobs
    stimulus_paths = collections.OrderedDict(sorted(stimulus_paths.items(), key=lambda x: x[0]))
    for stimulus_name, stimulus_local_path in tqdm(stimulus_paths.items(), desc='Processing stories'):
        wav, sample_rate = torchaudio.load(stimulus_local_path)
        if not args.resample:
            # Perform checks on the original waveform
            assert wav.shape[0] == 1, f"stimulus '{stimulus_local_path}' is not mono-channel"
            assert sample_rate == TARGET_SAMPLE_RATE
        else:
            # Resample & convert to mono as needed
            if wav.shape[0] != 1: wav = wav.mean(0, keepdims=True) # convert to mono
            if sample_rate != TARGET_SAMPLE_RATE: # resample to 16 kHz
                wav = torchaudio.functional.resample(wav, sample_rate, TARGET_SAMPLE_RATE)
                sample_rate = TARGET_SAMPLE_RATE

        wav.squeeze_(0) # shape: (num_samples,)

        assert sample_rate == TARGET_SAMPLE_RATE, f"Expected sample rate {TARGET_SAMPLE_RATE} but got {sample_rate}"

        features_save_path = os.path.join(model_save_path, stimulus_name)
        times_save_path = f"{features_save_path}_times"
        if not args.overwrite:
            if cci_features is None:
                if os.path.exists(times_save_path + '.npz'):
                    print(f"Skipping {stimulus_name}, timestamps found at {times_save_path}")
                    continue
            else:
                if cci_features.exists_object(times_save_path):
                    print(f"Skipping {stimulus_name}, timestamps found at {times_save_path}")
                    continue

        # Call a separate function to do the actual feature extraction
        extract_features_kwargs = {
            'model': model, 'model_config': model_config,
            'wav': wav.to(model.device),
            'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
            'sel_layers': args.layers, 'feature_extractor': feature_extractor,
            'require_full_context': args.full_context or args.pad_silence,
            'batchsz': args.batchsz, 'return_numpy': False
        }

        if args.stride:
            # Set the context_sz so that the total span length (context+chunk)
            # is the same as in non-stride mode, and so that the chunk_sz is
            # the "new" stride length.
            extract_features_kwargs['contextsz_sec'] = chunksz_sec + contextsz_sec - args.stride
            extract_features_kwargs['chunksz_sec'] = args.stride

        if args.pad_silence:
            # Pad with `context_sz` sec. of silence, so that the first
            # (non-silence) output is at time `chunk_sz`
            wav = torch.cat([torch.zeros(int(extract_features_kwargs['contextsz_sec']*TARGET_SAMPLE_RATE)), wav], axis=0)
            extract_features_kwargs['wav'] = wav.to(model.device)

        extracted_features = extract_features_hf(**extract_features_kwargs)
        out_features, times, module_features = [extracted_features[k] for k in \
                                                ['final_outputs', 'times', 'module_features']]
        del extracted_features # free up some memory after we've selected the outputs we want; maybe unnecessary

        # Remove the 'silence' we added at the beginning
        if args.pad_silence:
            times = torch.clip(times - extract_features_kwargs['contextsz_sec'], 0, torch.inf)
            assert torch.all(times >= 0), "padding is smaller than the correction (subtraction)!"
            assert torch.all(times[:,1] > 0), f"insufficient padding for require_full_context ! (times[times[:,1]<=0,1])"

        if cci_features is None:
            # If cottoncandy unavailable, save locally.
            os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
            np.savez_compressed(features_save_path + '.npz', features=out_features.numpy())
            np.savez_compressed(times_save_path + '.npz', times=times.numpy())
        else:
            cci_features.upload_raw_array(features_save_path, out_features.numpy())
            cci_features.upload_raw_array(times_save_path, times.numpy())

        module_save_paths = {module: os.path.join(model_save_path, module, stimulus_name) for module in module_features.keys()}

        # This is the "save name" of the module (not its original name)
        for module_name, features in module_features.items():
            features_save_path = module_save_paths[module_name]
            times_save_path = f"{features_save_path}_times"
            if cci_features is None:
                os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
                np.savez_compressed(features_save_path + '.npz', features=features.numpy())
                # "times" should be the same for all modules
            else:
                cci_features.upload_raw_array(features_save_path, features.numpy())
