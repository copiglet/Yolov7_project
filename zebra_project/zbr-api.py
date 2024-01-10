'''
file : zbr-api.py

python zbr-api.py --port 8080
ex) http://localhost:8080/api/v1/detect?dir=directory

'''
import argparse
import datetime, time
from pathlib import Path
import os
import sys
import shutil
import json
import csv
from bottle import route, request, response, abort, run, redirect
import io


parser = argparse.ArgumentParser()

def readLines(fn):
  with open(fn, encoding='utf-8') as fr:
    return [line.rstrip('\n') for line in fr]

def read_csv(csvFilePath, pkey):
    data = {"result": []}

    lines = readLines(csvFilePath)
    if not lines[0].startswith('File'):
      lines[0] = 'File' + lines[0]

    csvReader = csv.DictReader(io.StringIO('\n'.join(lines)))
    last_row = None
    for rows in csvReader:
      if (rows['File' if 'File' in rows else ''] == pkey):
        last_row = rows
    data["result"].append(last_row)
    return data

def clean_folder(folder):
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    print('clean:', file_path)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# read
def readLines(fn):
  with open(fn) as fr:
    return [line.rstrip('\n') for line in fr]

def getJsonResult(pkey):
  fn = os.path.join('output_json', f'{pkey}.json')
  with open(fn) as json_file:
    return json.load(json_file)

def makeFullJson(pkey):
  r = {"result": [{"File": pkey}]}
  x = r["result"][0]
  for row in range(0, 30):
    for col in range(0, 60):
      x[f'{row:02}-{col:02}.jpg'] = "None:0,0,0,0:0"

  for i in getJsonResult(pkey):
    x[i['img']] = i['result']

  return r

def docker_run():
  cmd = 'python docker_run.py'
  print(cmd)
  os.system(cmd)

@route('/api/ping')
def ping():
    response.content_type = 'application/json'
    return json.dumps({"pong": datetime.datetime.now().strftime("%Y%m%d%H%M%S")})


# ex) /api/v1/detect?dir=c:\data\2023-01-10\aaaaaaa
@route('/api/v1/detect')
def detect():
    p_dir = request.query.dir
    print(p_dir)

    if 'test' in request.query:
      print('test:', request.query.test)
      response.content_type = 'application/json'
      return json.dumps(read_csv('output.csv', request.query.test))
      # return json.dumps(makeFullJson(request.query.test))
    else:
      p_dir = os.path.normpath(p_dir)
      if os.path.isdir(p_dir):
        d_key = p_dir.split(os.sep)[-1]
        dst_dir = os.path.join('input', d_key)
        print("dest dir", dst_dir)

        clean_folder('input')

        shutil.copytree(p_dir, dst_dir)
        docker_run()
        response.content_type = 'application/json'
        return json.dumps(read_csv('output.csv', d_key))
        # return json.dumps(makeFullJson(d_key))
      else:
        print("path not found:", p_dir)
        abort(403, 'param fail, directory not found')


if __name__ == "__main__":
  print("")
  print("Zebra A.I. Detector : v0.1.2023.08.30")
  print("")
  parser.add_argument("-host", "--host", type=str,
                    help="bind address", default="0.0.0.0")
  parser.add_argument("-p", "--port", type=int,
                      help="tcp port", default=9888)
  parser.add_argument(
      "-csvtest", "--csvtest", help="csv test mode", action='store_true')
  parser.add_argument("-testid", "--testid", type=str,
                      help="testid for test", default="")
  parser.add_argument(
      "-d", "--debug", help="add debug mode", action='store_true')
  parser.add_argument(
      "-reload", "--reloader", help="debug reloader mode", action='store_true')
  parser.add_argument("-path", "--prgpath", type=str,
                      help="program path", default=os.getcwd())
  args = parser.parse_args()
  print(args)

  if args.csvtest and args.testid:
    #
    # python zebra-api.py --csvtest --testid=202204021712A
    #
    print(json.dumps(read_csv('output.csv', args.testid)))
  else:
    run(host="localhost", port=args.port, debug=args.debug, reloader=args.reloader)