import os
from datetime import datetime

def maybe_download(source_url, filename, target_directory, filetype='folder', force=False):
  """Download the data from some website, unless it's already here."""
  if source_url==None or filename==None: return
  if target_directory==None: target_directory = os.getcwd()
  filepath = os.path.join(target_directory, filename)
  if os.path.exists(filepath) and not force:
    print(filepath+' already exists, skipping download')
  else:
    if not os.path.exists(target_directory):
      os.system('mkdir -p '+target_directory)
    if filetype=='folder':
      os.system('curl -L '+source_url+' > '+filename+'.zip')
      os.system('unzip -o '+filename+'.zip'+' -d '+filepath)
      os.system('rm '+filename+'.zip')
    elif filetype=='tar':
      os.system('curl -o '+filepath+'.tar '+source_url)
      os.system('tar xzvf '+filepath+'.tar --directory '+target_directory)
      os.system('rm '+filepath+'.tar')
    elif filetype=='file':
      os.system('curl -L -o '+filepath+' '+source_url)

def timenow():
  return datetime.now().strftime('%m-%d_%H:%M:%S_%f')
