import glob
import csv
import librosa
import sys


endings = "[m4a|opus|wav|mp3]"

def get_entries(directory):
    files = glob.glob(f"./{directory}/*.{endings}")
    entries = []
    for f in files:
        try:
            entries.append((directory, f, librosa.get_duration(f)))
        except:
            raise
    return entries



def main():
    all_entries = []
    for dir_name in sys.argv[1:]:
        all_entries += get_entries(dir_name)
    with open("out.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "location", "duration"])
        writer.writerows(all_entries)

if __name__ == "__main__":
    main()
