from service import svc

for runner in svc.runners:
    runner.init_local()


result = svc.apis["sv_extract"].func("CVE Description")

print(result)