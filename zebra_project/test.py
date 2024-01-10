import pandas as pd
import csv
import io

# csv = pd.read_csv('output1.csv', index_col=0)
 
# index_ls = csv.index

# index_ls = set(index_ls)
# index_ls = list(index_ls)
# index_ls.sort()

# print(csv.columns)

# 
# csv.to_csv('outpindex_lsut1.csv')


def readLines(fn):
  with open(fn, encoding='utf-8') as fr:
    return [line.rstrip('\n') for line in fr]

def read_csv(csvFilePath):
  lines = readLines(csvFilePath)
  if not lines[0].startswith('File'):
    lines[0] = 'File' + lines[0]

  csvReader = csv.DictReader(io.StringIO('\n'.join(lines)))
  for rows in csvReader:
      if (rows['File' if 'File' in rows else ''] == '202204021712A'):
        print(rows['00-00.jpg'])
        break

read_csv('output1.csv')
