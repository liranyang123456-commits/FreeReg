import fitz
pdf = fitz.open('freereg_tbme_full.pdf')
for pi in range(4, 10):
    p = pdf[pi]
    infos = p.get_image_info()
    print(f'--- page {pi+1}: {len(infos)} images')
    for i in infos:
        b = i['bbox']
        print(f"  w={b[2]-b[0]:6.0f} h={b[3]-b[1]:6.0f} x0={b[0]:5.0f} y0={b[1]:5.0f}")
