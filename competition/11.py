from PIL import Image

# 1. 打开原始图像（替换为你的图像路径）
original_img = Image.open("111.png")
width, height = original_img.size

# 2. 定义白色边框的宽度（可根据需要调整，单位：像素）
border_width = 200 
border_height = 200

# 3. 创建新图像：尺寸为原尺寸 + 2×边框宽度，背景色为白色
new_width = width + 2 * border_width
new_height = height + 2 * border_height
new_img = Image.new("RGB", (new_width, new_height), color="white")

# 4. 将原图像粘贴到新图像的中间位置（四周留出边框宽度）
new_img.paste(original_img, (border_width, border_width))

# 5. 保存带白色边框的新图像（可自定义输出路径）
new_img.save("comap_with_white_border.png")