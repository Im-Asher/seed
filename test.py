from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "task":"sv",
    "samples":["sockethandler.cpp in HTTP Antivirus Proxy (HAVP) 0.88 allows remote attackers to cause a denial of service (hang) by connecting to a non-responsive server, which triggers an infinite loop due to an uninitialized variable. c++"]
}

result = svc.apis["extract"].func(request)

print(result)
