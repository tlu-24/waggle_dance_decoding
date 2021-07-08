configfile:"config.yaml"


rule scale_down:
    input:
        mp4="{video_label}.MP4",
        res="{resolution}"
    output:
        "{video_label}_{resolution}.MP4"
    shell:
        "ffmpeg -i {input.mp4} -vf scale=-1:{input.res} {output}"

# rule split_manifest:
#     input:
#         mp4="{video_label}.MP4"
#         json="{manifest}.json"
#     output:

rule scale_calibrate:
    input:
        mp4="{video_label}.MP4",
        out="{out_dir}",
        left="{left}"
    output:
        "{video_label}-scale.pkl"
    shell:
    # what is going on with the python version
        "python3.7 scale_calibrate.py -i {input.mp4} -o {input.out} -l {input.left}"

rule dance_detect:
    input:
        mp4="{video_label}.MP4",
        pkl="{video_label}-scale.pkl"
    output:
        "{video_label}-Waggle_Detections.pkl"
    shell:
        "python3.7 DanceDetector.py -i {input.mp4} -c {input.pkl}"

rule eps_cluster:
    input:
        "{video_label}-Waggle_Detections.pkl"
    output:
        "{video_label}-findepscluster_{cluster_params}.pkl"
    shell:
        "python3.7 find_eps.py -i {input}"