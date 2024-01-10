import os

PATH = os.getcwd()

output_csv = "output.csv"
if os.path.exists(output_csv):
    if os.path.isdir(output_csv):
        os.rmdir(output_csv)
if not os.path.exists(output_csv):
    os.system(f"copy output_empty.csv {output_csv}")

# PATH = "/mnt/nas03/phenomx/huvio/mnt/sata4t/phenomx/huvio/zebra/zebra_project2"
cmd = 'docker run --rm -it \
                --gpus=all \
                --name zebra_final \
                -v {0}/detection.py:/workspace/detection.py:rw \
                -v {0}/utils:/workspace/utils:rw \
                -v {0}/input:/workspace/input:rw \
                -v {0}/cropped_egg:/workspace/cropped_egg:rw \
                -v {0}/detect_infer:/workspace/detect_infer:rw \
                -v {0}/final_larva:/workspace/final_larva:rw \
                -v {0}/output.csv:/workspace/output.csv:rw \
                -v {0}/run_script:/workspace/run_script:rw \
                -v {0}/output_json:/workspace/output_json:rw \
                -v {0}/run.py:/workspace/run.py:rw \
                zebra:v2'.format(PATH)
os.system(cmd)

# import os

# PATH = os.getcwd()

# output_csv = "output.csv"
# if os.path.exists(output_csv):
#     if os.path.isdir(output_csv):
#         os.rmdir(output_csv)
# if not os.path.exists(output_csv):
#     os.system(f"copy output_empty.csv {output_csv}")
    
# cmd = 'docker run --rm -it \
#                 --gpus=all \
#                 --name zebra_test \
#                 -v {0}/input:/workspace/input:rw \
#                 -v {0}/cropped_egg:/workspace/cropped_egg:rw \
#                 -v {0}/detect_infer:/workspace/detect_infer:rw \
#                 -v {0}/final_larva:/workspace/final_larva:rw \
#                 -v {0}/output.csv:/workspace/output.csv:rw \
#                 -v {0}/run_script:/workspace/run_script:rw \
#                 -v {0}/output_json:/workspace/output_json:rw \
#                 -v {0}/run.py:/workspace/run.py:rw \
#                 zebra:v1'.format(PATH)
# os.system(cmd)