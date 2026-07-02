import pathlib, re

pylidc_dir = pathlib.Path(r'C:\Users\emkap\Documents\Projects\LungInsight\.venv\Lib\site-packages\pylidc')

replacements = [
    (r'np\.int\b',   'int'),
    (r'np\.float\b', 'float'),
    (r'np\.bool\b',  'bool'),
]

for fname in ('Contour.py', 'Annotation.py', 'utils.py'):
    p = pylidc_dir / fname
    text = p.read_text(encoding='utf-8')
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    p.write_text(text, encoding='utf-8')
    print('Patched:', fname)

scan_py = pylidc_dir / 'Scan.py'
text = scan_py.read_text(encoding='utf-8')
text = text.replace('configparser.SafeConfigParser()', 'configparser.ConfigParser()')
scan_py.write_text(text, encoding='utf-8')
print('Patched: Scan.py')
print('Done.')
