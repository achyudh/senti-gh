from lib.data import collate, download
from lib.util import db
import os.path



def repo_reaper(dataset_path, num_pages=10, skip_existing=False, is_list=False):
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


def custom_annotated(dataset_path, full_repo_name, num_pages=10):
    with open(dataset_path, 'r') as dataset_csv:
        _first_line = dataset_csv.readline()
        repo_data = dict()
        for line in dataset_csv:
            pr_data = dict()
            pr_comments = list()
            line_split = line.strip().split(',')
            pr_number = int(line_split[0].split('/')[-1])
            pr_info = download.pull_requests(full_repo_name, num_pages, pr_number=pr_number)[0]
            pr_comments.append({"user": pr_info["user"], "body": pr_info["body"], "pr_body": True,
                                "reactions": {"total_count": 0, "+1": 0, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 0},})
            pr_comments.extend(download.issue_comments(full_repo_name, num_pages, issue_number=pr_number))
            if len(pr_comments) != int(line_split[2]):
                print(line_split[0], len(pr_comments), line_split[2])
            else:
                ctr = 0
                for pr_comment in pr_comments:
                    pr_comment['ipa'] = line_split[1][ctr]
                    ctr += 1
            pr_data['comments'] = pr_comments
            pr_data['info'] = pr_info
            repo_data[pr_number] = pr_data
    db.insert_generic(repo_data, dataset_path.split('.')[0] + ".json")


def custom_list(repo_list, num_pages=10, skip_existing=False):
    for full_repo_name in repo_list:
        if not skip_existing or not(os.path.isfile('data/user/%s.json' % full_repo_name) and os.path.isfile('data/repo/%s.json' % full_repo_name)):
            print("Fetching %s..." % full_repo_name)
            collate.by_user(full_repo_name, num_pages)



if __name__ == '__main__':
    # custom_list(repo_list=["tensorflow/tensorflow", "onivim/oni", "spring-projects/spring-framework"])
    custom_annotated('data/labelled/pull_requests/oni.csv', full_repo_name="onivim/oni")
    # repo_reaper('data/repo_reaper/reaper_100.csv', num_pages=10, skip_existing=True, is_list=True)
    # repo_reaper('data/repo_reaper/utility.csv', num_pages=10, skip_existing=True)
    # repo_reaper('data/repo_reaper/validation.csv', num_pages=10, skip_existing=True)