from unrar import rarfile

file = rarfile.RarFile('data.rar')
file.extractall('./')
