import glob
import csv
import librosa
import sys
import os


endings = ["m4a", "opus", "wav", "mp3"]

def get_entries(directory):
    entries = []
    for end in endings:
        files = glob.glob(os.path.join(directory, f"*.{end}"))
        for f in files:
            try:
                entries.append((directory, f, librosa(f)))
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
