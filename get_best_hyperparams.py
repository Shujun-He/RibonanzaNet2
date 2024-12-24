from glob import glob

best_score=0
for f in glob("grid_search_scores/*"):
    with open(f,'r') as file:
        score=float(file.read())
    #print(score)
    if score>best_score:
        best_score=score
        best_config=f

print(best_score)
print(best_config)
