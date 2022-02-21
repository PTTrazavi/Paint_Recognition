import os

for i, dirs in enumerate(os.listdir('./20220107JURASSIC_PIC')):

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/左上'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/左上', f'./20220107JURASSIC_PIC/{dirs}/01')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/右上'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/右上', f'./20220107JURASSIC_PIC/{dirs}/02')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/左下'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/左下', f'./20220107JURASSIC_PIC/{dirs}/03')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/右下'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/右下', f'./20220107JURASSIC_PIC/{dirs}/04')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/Ａ'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/Ａ', f'./20220107JURASSIC_PIC/{dirs}/01')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/Ｂ'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/Ｂ', f'./20220107JURASSIC_PIC/{dirs}/02')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/Ｃ'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/Ｃ', f'./20220107JURASSIC_PIC/{dirs}/03')

	if os.path.isdir(f'./20220107JURASSIC_PIC/{dirs}/Ｄ'):
		os.rename(f'./20220107JURASSIC_PIC/{dirs}/Ｄ', f'./20220107JURASSIC_PIC/{dirs}/04')

	os.rename(f'./20220107JURASSIC_PIC/{dirs}', f'./20220107JURASSIC_PIC/{i:04d}')
