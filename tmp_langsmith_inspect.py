import importlib
import inspect

langsmith = importlib.import_module('langsmith')
print('langsmith', getattr(langsmith, '__version__', None))
client = importlib.import_module('langsmith.client').Client
print('Client', client)
print('Signature:', inspect.signature(client))
print('Client methods', [name for name in dir(client) if 'create' in name.lower() or 'run' in name.lower() or 'span' in name.lower()])
