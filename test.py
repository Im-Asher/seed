from service import svc

for runner in svc.runners:
    runner.init_local()
request = {
    "task":"sv",
    "samples":["In lg_probe and related functions of hid-lg.c and other USB HID files,\
                there is a possible out of bounds read due to improper input validation. \
               This could lead to local information disclosure if a malicious USB HID device were plugged in,\
                with no additional execution privileges needed. User interaction is not needed for exploitation.\
               Product: AndroidVersions: Android kernelAndroid ID: A-188677105References: Upstream kernel",
               "Cross-site Scripting (XSS) - Stored in Packagist pimcore/pimcore prior to 10.2.9."]
}

result = svc.apis["extract"].func(request)

print(result)