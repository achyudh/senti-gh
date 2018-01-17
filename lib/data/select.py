from lib.data import collate


def repo_reaper(dataset_path):
    with open(dataset_path, 'r') as dataset_csv:
        _first_line = dataset_csv.readline()
        for line in dataset_csv:
            full_repo_name = '/'.join(line.split(',')[0].split('/')[3:5])
            collate.by_user(full_repo_name)


if __name__ == '__main__':
    repo_reaper('data/repo_reaper/organization.csv')