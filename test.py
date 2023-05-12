from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "task":"sv",
    "samples":["OpenHarmony-v3.1.2 and prior versions have a heap overflow vulnerability. Local attackers can trigger a heap overflow and get network sensitive information."]
}

result = svc.apis["extract"].func(request)

print(result)
