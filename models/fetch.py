import pickle5 as pickle
import hashlib,tempfile,requests,os

def fetch(url):
  if url.startswith("/"):
    with open(url,"rb") as file:
      data = file.read()
      return data

  #filepath
  fp = os.path.join(tempfile.gettempdir(),hashlib.md5(url.encode("utf-8")).hexdigest())

  if os.path.isfile(fp) and os.stat(fp).st_size > 0 and os.getenv("NOCACHE",None) is None:
    with open(fp, "rb") as f:
      data = f.read()
  else:
    print("Fetching url %s" %url)
    r = requests.get(url)
    assert r.status_code == 200
    data = r.content
    with open(fp+".tmp","wb") as f:
      f.write(data)
    os.rename(fp+".tmp",fp)

  return data
