import inspect
import os
from langsmith.client import Client

print('Client signature:', inspect.signature(Client))
print('Client methods:', [name for name in dir(Client) if 'run' in name.lower() or 'span' in name.lower() or 'create' in name.lower()])

client = Client(api_key=os.environ.get('LANGSMITH_API_KEY'), web_url='https://api.langsmith.com')
print('Client instance methods:', [name for name in dir(client) if 'run' in name.lower() or 'span' in name.lower() or 'create' in name.lower()])

try:
    run = client.create_run(name='test-tracing-run', project=os.environ.get('LANGSMITH_PROJECT', 'multi-agent-research-lab'))
    print('run type', type(run))
    print('run attrs:', [name for name in dir(run) if 'id' in name.lower() or 'url' in name.lower() or 'create' in name.lower() or 'span' in name.lower() or 'run' in name.lower()][:100])
    print('run repr', repr(run))
except Exception as exc:
    print('create_run failed', exc)
