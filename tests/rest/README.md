# REST contract tests

These tests start only local loopback HTTP servers. They never call an object webhook,
change `detector_config.json`, or access the network outside the test process.

Run them separately:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests\rest -p "test_*.py" -v
```
