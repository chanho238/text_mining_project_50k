# 50k v12 template diversity and shortcut repair

v11 is the accepted baseline. v12 is a patch focused on template-level similarity, shortcut AUC, SVM false positives, and enterprise/user distribution clarity.

Normalized duplicate is separated from true leakage. v12 reduces high-risk template repetition while preserving validation/test rows unless a concrete error exists.

SVM FP rows from v11 were normal questions, so they were not removed or relabeled. Similar normal support was added to train rows.

Final decision: v12 accepted. Release: PASS, preferred: FAIL, strong: FAIL.
