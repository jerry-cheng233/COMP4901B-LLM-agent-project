"""Check what pages/databases the Notion API can access."""
import sys
sys.path.insert(0, 'src')
from notion import Notion

NOTION_API_KEY = "YOUR_NOTION_API_KEY_HERE" 

n = Notion(NOTION_API_KEY)
results = n.search('')

print('=' * 50)
print('Pages/Databases accessible to Notion API:')
print('=' * 50)

for r in results.get('results', []):
    obj_type = r.get('object', '')
    if obj_type == 'database':
        title_list = r.get('title', [])
        title = title_list[0].get('plain_text', 'Untitled') if title_list else 'Untitled'
    else:
        props = r.get('properties', {})
        # Try different title property names
        for key in ['title', 'Name', 'Task name']:
            if key in props and 'title' in props[key]:
                title_arr = props[key]['title']
                title = title_arr[0]['plain_text'] if title_arr else 'Untitled'
                break
        else:
            title = 'Untitled'
    
    print(f'  [{obj_type:8}] {title}')

print()
print('If your notes are missing, share them with the integration:')
print('  1. Open the page in Notion')
print('  2. Click ••• menu > Connections > Add your integration')

