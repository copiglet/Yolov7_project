import subprocess
import os
# th = '0.95,0.8,0.8,0.8,0.95,0.8,0.8,0.8,0.8,0.8'
path = '/home/huvio/Project/huvio/data/ag_data/2023/03/15/m'
for i in os.listdir(path):
    print(i)
    process = subprocess.run(['bash', '/home/huvio/Project/huvio/ai_part/batch_test.sh',i], check=True)

