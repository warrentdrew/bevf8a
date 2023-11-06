# !/usr/bin/env python3
"""
Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

"""
Exports a video of each scene (with annotations) to disk.
"""

import argparse
import os

from nuscenes import NuScenes


def export_videos(nusc: NuScenes, out_dir: str):
    """ Export videos of the images displayed in the images. """

    # Load NuScenes class
    scene_tokens = [s['token'] for s in nusc.scene]

    # Create output directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Write videos to disk
    for scene_token in scene_tokens:
        scene = nusc.get('scene', scene_token)
        print('Writing scene %s' % scene['name'])
        out_path = os.path.join(out_dir, scene['name']) + '.avi'
        if not os.path.exists(out_path):
            nusc.render_scene(scene['token'], out_path=out_path)


if __name__ == '__main__':

    # Settings.
    parser = argparse.ArgumentParser(description='Export all videos of annotations.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_dir', type=str, help='Directory where to save videos.', default='videos')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')

    args = parser.parse_args()
    dataroot = args.dataroot
    version = args.version
    verbose = bool(args.verbose)

    # Init.
    nusc_ = NuScenes(version=version, verbose=verbose, dataroot=dataroot)

    # Export videos of annotations
    export_videos(nusc_, args.out_dir)
