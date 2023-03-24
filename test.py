from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "task":"sv",
    "samples":["An issue has been discovered in GitLab affecting all versions, GitLab versions starting from 9.0 before 15.7.8, GitLab versions starting from 15.8 before 15.8.4, all versions starting from 15.9 before 15.9.2. It was possible to trigger a resource depletion attack due to improper filtering for number of requests to read commits details."]
}

result = svc.apis["extract"].func(request)

print(result)
