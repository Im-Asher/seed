from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "task":"sv",
    "samples":["GitLab EE 11.3 through 13.1.2 has Incorrect Access Control because of the Maven package upload endpoint."]
}

result = svc.apis["extract"].func(request)

print(result)
