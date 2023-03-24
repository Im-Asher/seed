from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "task":"sv",
    "samples":["LibTIFF 4.4.0 has an out-of-bounds write in tiffcrop in tools/tiffcrop.c:3724, allowing attackers to cause a denial-of-service via a crafted tiff file. For users that compile libtiff from sources, the fix is available with commit 33aee127."]
}

result = svc.apis["extract"].func(request)

print(result)
