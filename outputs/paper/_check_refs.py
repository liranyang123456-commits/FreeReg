import re
tex = open(r'E:\Free_coordinate_MR_Registration\outputs\paper\freereg_tbme_full.tex', encoding='utf-8').read()

# Key numbers to check for consistency
checks = {
    '4.23': 'clinical TRE mean',
    '2.93': 'clinical TRE median',
    '92\\%': 'under 10mm',
    '5--10': 'ENB range',
    '104': 'FoundationPose fail',
    '34': 'Endo3R ATE',
    '46.3': 'PSNR',
    '112': 'FPS render',
    '151/648': 'observability rank',
    '23\\%': 'observable fraction',
    '1.44': 'single-shot time',
    '30--80': 'clinical FPS',
    '0.848': 'BOP AR',
    '1.1\\%': 'real-colon recovery',
}
print('=== number consistency ===')
for num, desc in checks.items():
    n = len(re.findall(re.escape(num), tex))
    flag = 'OK' if n >= 1 else 'MISSING'
    print(f'  [{flag}] {desc}: "{num}" x{n}')

# Section structure
print('\n=== sections ===')
for m in re.findall(r'\\section\{([^}]+)\}', tex):
    print('  ', m)
print('\n=== subsections ===')
for m in re.findall(r'\\subsection\{([^}]+)\}', tex):
    print('  ', m)
