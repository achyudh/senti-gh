from lib.data import collate
import os.path


def repo_reaper(dataset_path, num_pages=1, skip_existing=False, is_list=False):
    with open(dataset_path, 'r') as dataset_csv:
        _first_line = dataset_csv.readline()
        for line in dataset_csv:
            if not is_list:
                full_repo_name = '/'.join(line.split(',')[0].split('/')[3:5])
            else:
                full_repo_name  = line.split(',')[1].strip()
            if not skip_existing or not(os.path.isfile('data/user/%s.json' % full_repo_name) and os.path.isfile('data/repo/%s.json' % full_repo_name)):
                print("Fetching %s..." % full_repo_name)
                collate.by_user(full_repo_name, num_pages)


if __name__ == '__main__':
    repo_reaper('data/repo_reaper/reaper_2000.csv', num_pages=10, skip_existing=True, is_list=True)
    # repo_reaper('data/repo_reaper/utility.csv', num_pages=10, skip_existing=True)
    # repo_reaper('data/repo_reaper/validation.csv', num_pages=10, skip_existing=True)