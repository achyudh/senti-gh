from lib.data import collate
import os.path


def repo_reaper(dataset_path, skip_existing=False):
    with open(dataset_path, 'r') as dataset_csv:
        _first_line = dataset_csv.readline()
        for line in dataset_csv:
            full_repo_name = '/'.join(line.split(',')[0].split('/')[3:5])
            if skip_existing and not(os.path.isfile('data/user/%s.json' % full_repo_name) and os.path.isfile('data/repo/%s.json' % full_repo_name)):
                print("Fetching %s..." % full_repo_name)
                collate.by_user(full_repo_name)


if __name__ == '__main__':
    repo_reaper('data/repo_reaper/organization.csv', skip_existing=True)