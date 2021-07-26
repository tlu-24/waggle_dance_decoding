# split.py
# adapted with slight modification from https://github.com/beelabhmc/ant_tracker/blob/master/scripts/split.py
# Thanks Jarred
# Wrapper file for calls to ffmpeg to help with splitting of videos

import csv
import subprocess
import re
import math
import json
import os
import os.path
import argparse
import datetime
import pandas as pd


re_length = re.compile(r'Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,')

# extract important info from the filename


def get_global_start(filename):
    """ 
    extracts information about the colony, date, time and videoname from
    a filename of a long bee video.
        Inputs:
            - video filename
        Outputs: 
            - colony, date/time, videoname from the videofilename
    """
    betterfilename = filename.split('/')
    filename = betterfilename[-1]
    (colony, date, time, videoname) = filename.split('_')
    print("date", int(date) - (int(date) // 100 * 100),
          int(date)//10000, (int(date) - (int(date)//10000 * 10000))//100)
    outdate = datetime.datetime(2000 + int(date) - (int(date) // 100 * 100),
                                int(date)//10000, (int(date) - (int(date)//10000 * 10000))//100, int(time)//100, int(time)-(int(time)//100 * 100))
    outtime = datetime.time(int(time)//100, int(time)-(int(time)//100 * 100))
    print(colony, outdate, outtime, videoname)
    return (colony, outdate, videoname)


def date_to_string(datetime):
    """ 
    creates a string from a datetime, for use in filenames
        Inputs:
            - datetime
        Outputs: 
            - datetime in a string as "mmddyyyy_hhmmss"
    """
    y = str(int(datetime.year))
    m = str(int(datetime.month))
    d = str(int(datetime.day))
    h = str(int(datetime.hour))
    m = str(int(datetime.minute))
    s = str(int(datetime.second))
    return m+d+y+'_'+h+m+s


def by_manifest(filename, destination, manifest, vcodec='copy', acodec='copy',
                extra='', **kwargs):
    """ Split video into segments based on the given manifest file.
    Arguments:
        filename (str)      - Location of the video.
        destination (str)   - Location to place the output videos
                              (doesn't actually work because I don't know how
                               the manifest works --Jarred)
        manifest (str)      - Location of the manifest file.
        vcodec (str)        - Controls the video codec for the ffmpeg video
                              output.
        acodec (str)        - Controls the audio codec for the ffmpeg video
                              output.
        extra (str)         - Extra options for ffmpeg.
    """
    if not os.path.exists(manifest):
        print('File does not exist:', manifest)
        raise SystemExit

    outdict = {'name': [], 'start_time': [], 'length': []}
    (colony, date, videoname) = get_global_start(filename)
    current_start = date

    with open(manifest) as manifest_file:
        manifest_type = manifest.split('.')[-1]
        if manifest_type == 'json':
            config = json.load(manifest_file)
        elif manifest_type == 'csv':
            config = csv.DictReader(manifest_file)
        else:
            print('Format not supported. File must be a csv or json file')
            raise SystemExit

        # split_cmd = 'ffmpeg -loglevel warning -i \'{}\' -vcodec {} ' \
        #             '-acodec {} -y {}'.format(filename, vcodec, acodec, extra)
        split_cmd = 'ffmpeg -loglevel warning -i \'{}\' -vcodec {} ' \
                    '-an -y {} '.format(
                        filename, vcodec, extra)
        split_count = 1
        split_error = []
        try:
            fileext = filename.split('.')[-1]
        except IndexError as e:
            raise IndexError('No . in filename. Error: ' + str(e))
        for video_config in config:
            split_str = ''
            try:
                split_start = video_config['start_time']
                split_length = video_config.get('end_time', None)
                outdict['name'].append(video_config.get('rename_to', None))
                current_start = date + datetime.timedelta(
                    seconds=split_start)
                outdict['start_time'].append(current_start)
                outdict['length'].append(split_start)

                if not split_length:
                    split_length = video_config['length']
                filebase = video_config['rename_to']
                if fileext in filebase:
                    filebase = '.'.join(filebase.split('.')[:-1])

                split_str = ' -ss {} -t {} -avoid_negative_ts 1 "{}.{}"' \
                            .format(split_start, split_length, filebase, fileext)
                better_split_cmd = 'ffmpeg -loglevel warning -ss {} -i {} -t {} -vcodec {} -an -y "{}.{}"'.format(
                    split_start, filename, split_length, vcodec, filebase, fileext)
                print('#######################################################')
                # print('About to run: '+split_cmd+split_str)
                print('About to run:', better_split_cmd)
                print('#######################################################')
                # output = subprocess.Popen(split_cmd+split_str,
                #   shell=True, stdout=subprocess.PIPE).stdout.read()
                output = subprocess.Popen(better_split_cmd,
                                          shell=True, stdout=subprocess.PIPE).stdout.read()
            except KeyError as e:
                print('############# Incorrect format ##############')
                if manifest_type == 'json':
                    print('The format of each json array should be:')
                    print('{start_time: <int>, length: <int>, rename_to: '
                          '<string>}')
                elif manifest_type == 'csv':
                    print('start_time,length,rename_to should be the first '
                          'line ')
                    print('in the csv file.')
                print('#############################################')
                print(e)
                raise SystemExit
        out_df = pd.DataFrame(outdict)
        print(out_df)
        out_df.to_pickle(destination + str(colony)+'_' +
                         date_to_string(date)+'_'+videoname + '.pkl')


def get_video_duration(input_video):

    duration_regex = re.compile(r'^duration=([0-9]+[.][0-9]+)$', re.MULTILINE)
    """Returns the duration of the given video."""
    cmd = f'ffprobe -show_streams {input_video}'.split()
    output = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT).stdout.decode()
    try:
        return float(re.search(duration_regex, output).group(1))
    except AttributeError as e:
        raise RuntimeError('Unparsable ffprobe output:\n{}\nfrom command:\n{}'
                           .format(output, ' '.join(cmd))) from e


def by_seconds(filename, destination, split_length, vcodec='copy',
               acodec='copy', extra='', min_segment_length=20, **kwargs):
    if not os.path.isdir(destination):
        os.makedirs(destination)
    if split_length <= 0:
        print('Split length must be positive')
        raise SystemExit
    try:
        video_length = get_video_duration(filename)
    except RuntimeError as re:
        print("Can't determine video length, copying video without splitting")
        destination = os.path.join(destination, '0.mp4')
        subprocess.Popen(f'cp "{filename}" "{destination}"', shell=True)
        return
    split_count = math.ceil(video_length / split_length)

    if(split_count == 1):
        print('Video length is less then the target split length.')
        destination = os.path.join(destination, '0.mp4')
        subprocess.Popen(f'cp "{filename}" "{destination}"', shell=True)
        return

    outdict = {'name': [], 'start_time': [], 'length': []}
    (colony, date, videoname) = get_global_start(filename)
    current_start = date
    # we use -y to force overwrites of output files
    # split_cmd = f'ffmpeg -loglevel warning -y -i \'{filename}\' -vcodec ' \
    #             f'{vcodec} -acodec {acodec} {extra}'
    split_cmd = f'ffmpeg -loglevel warning -vcodec ' \
                f'{vcodec} -acodec {acodec} {extra}'
    # get the filename without the file ext
    filebase = os.path.basename(filename)
    filebase, fileext = os.path.splitext(filebase)
    # for n in range(0, split_count):
    for n in range(0, 3):
        split_start = split_length * n

        current_start = date + datetime.timedelta(
            seconds=split_start)

        outdict['name'].append(colony+'_'+str(current_start)+'.MP4')
        outdict['start_time'].append(current_start)
        outdict['length'].append(split_length)

        if video_length - split_start < min_segment_length:
            print(f'Not copying the last {video_length-split_start} seconds.')
            continue
        # split_str = f' -ss {split_start} -t {split_length} {destination}{n}.mp4'
        split_str = f' -ss {split_start} -y -i \'{filename}\' -t {split_length} {destination}{n}.mp4'
        print('About to run:', split_cmd, split_str, sep='')
        # output = subprocess.Popen(split_cmd+split_str, shell=True,
        #                           stdout=subprocess.PIPE,
        #                           ).stdout.read()
        # better_split_cmd = 'ffmpeg -loglevel warning -ss {} -i {} -t {} -vcodec {} -an -y "{}{}{}"'.format(
        #     split_start, filename, split_length, vcodec, filebase, n, fileext)
        better_split_cmd = 'ffmpeg -loglevel warning -ss {} -i {} -t {} -vcodec {} -an -y "{}{}{}"'.format(
            split_start, filename, split_length, vcodec, filebase, colony+'_'+str(current_start)+'.MP4', fileext)
        output = subprocess.Popen(better_split_cmd,
                                  shell=True, stdout=subprocess.PIPE).stdout.read()
    out_df = pd.DataFrame(outdict)
    print(out_df)
    out_df.to_pickle(destination + str(colony)+'_' +
                     date_to_string(date)+'_'+videoname + '.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        type=str,
                        help='The name of the file to split',
                        )
    parser.add_argument('destination',
                        type=str,
                        help='The directory in which to save the split videos.',
                        )
    parser.add_argument('-s', '--split-size',
                        dest='split_length',
                        type=int,
                        help='Split or chunk size in seconds, for example 10',
                        )
    parser.add_argument('-m', '--manifest',
                        dest='manifest',
                        type=str,
                        help='Split video based on a json manifest file. ',
                        )
    parser.add_argument('-v', '--vcodec',
                        dest='vcodec',
                        type=str,
                        default='copy',
                        help='Video codec to use. If unspecified, it defaults '
                             'to the one in the source video.',
                        )
    parser.add_argument('-a', '--acodec',
                        dest='acodec',
                        type=str,
                        default='copy',
                        help='Audio codec to use. If unspecified, it defaults '
                             'to the one in the source video',
                        )
    parser.add_argument('-l', '--min-segment-length',
                        dest='min_segment_length',
                        type=float,
                        default=20,
                        help='The minimum length of a segment. If the last '
                             'segment of the video is shorter than this '
                             'length, then it is ignored. Default: 20')
    parser.add_argument('-e', '--extra',
                        dest='extra',
                        type=str,
                        default='',
                        help='Extra options for ffmpeg, e.g. "-e -threads 8". ',
                        )
    args = parser.parse_args()
    if args.destination[-1] != '/':
        args.destination += '/'

    if args.filename and args.manifest:
        by_manifest(**(args.__dict__))
    elif args.filename and args.split_length:
        by_seconds(**(args.__dict__))
    else:
        parser.print_help()
        raise SystemExit


if __name__ == '__main__':
    main()
