# MIT License
#
# Copyright (c) 2024 Keisuke Sehara and Ryo Aoki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO: load the dictionary below from e.g. a JSON file in the future...
CUED_LEVER_PULL = {
    'columns': [
        {
            'name': 'trial_start',
            'output_name': 'start_time',
            'description': 'the timing of the cue onset',
        },
        {
            'name': 'trial_end',
            'output_name': 'stop_time',
            'description': 'the timing when the trial outcome is determined',
        },
        {
            'name': 'pull_onset',
            'description': 'the timing when the animal started to pull the lever',
        },
        {
            'name': 'reaction_time',
            'description': "the time interval (in seconds) from the cue onset to when the animal started to pull the lever",
        },
        {
            'name': 'pull_duration_for_success',
            'description': 'the duration (in seconds) that the animal was required to pull the lever to obtain reward for the trial',
        },
        {
            'name': 'trial_outcome',
            'data_type': 'int',  # FIXME
            'description': 'the outcome of the trial',
        }
    ]
}

SENSORY_STIM = {
    'columns': [
        {
            'name': 'trial_start',
            'output_name': 'start_time',
            'description': 'the timing of the stimulus onset',
        },
        {
            'name': 'trial_end',
            'output_name': 'stop_time',
            'description': 'the timing of the stimulus offset',
        },
        {
            'name': 'stim_modality',
            'data_type': 'int',  # FIXME
            'description': "the modality of the stimulus: `visual (1)`, a flash of LED in front of the animal's eye; `auditory (2)`, a brief buzz of white noise from the speaker on the front-left side of the animal; `somatosensory (3)`, a brief vibration to the right whisker-pad of the animal"
        }
    ],
}
