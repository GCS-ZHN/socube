import os
import sys
import src.socube as sc
tag = os.environ["GITHUB_REF"].replace("refs/tags/v", "")
if (sc.__version__ != tag):
    print("Version mismatch:")
    print("  socube: " + sc.__version__)
    print("  repo tag: " + tag)
    sys.exit(1)