from text_indexing import generate_person_description, generate_project_description

# # Test person
# name_en = "Dr. Somchai Wichitpong"
# keywords = "longan,postharvest,grading"
# research_interests_th = "ลำไย, เทคโนโลยีหลังการเก็บเกี่ยว, มาตรฐานคัดเกรด"

# person_description = generate_person_description(name_en, keywords, research_interests_th)
# print("=== Person description ===")
# print(person_description)

# # Test project — using PROJ001 actual data from your graph
# title_en = "Development of Automated Longan Grading System Using Machine Vision to Reduce Downgrading"
# proj_keywords = "longan,grading,machine vision,AI,downgrade"
# abstract_th = "โครงการพัฒนาระบบคัดเกรดลำไยอัตโนมัติโดยใช้เทคโนโลยี Machine Vision และ AI เพื่อเพิ่มประสิทธิภาพการคัดเกรดและลดการสูญเสียจากการตกเกรด"
# acronym = "ALGO"

# project_description = generate_project_description(title_en, proj_keywords, abstract_th, acronym)
# print("\n=== Project description ===")
# print(project_description)

from vector_indexing import get_embedding

text = "Dr. Somchai Wichitpong specializes in the postharvest technologies of longan, focusing on the grading standards essential for quality assessment and market readiness."

vector = get_embedding(text)

print(f"Embedding length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")