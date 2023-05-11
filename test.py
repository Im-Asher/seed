from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "task":"sv",
    "samples":["versions 1.3.0 to 1.7.7 are vulnerable against stored XSS via the “Web Page” element, that allows the injection of malicious JavaScript into the 'URL' field. This issue affects: nasa openmct 1.7.7 version and prior versions; 1.3.0 version and later versions."]
}

result = svc.apis["extract"].func(request)

print(result)
