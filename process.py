import os

path = "D:/Segmentacja/MedSegDiff/results/"
files = os.listdir(path)

for file in files:
	patient = file[:file.find("_output")]
	
	if not os.path.exists(path + patient):
		os.makedirs(path + patient)
	
	os.rename(path + file, path + patient + "/" + file)

print("Ok!")
		