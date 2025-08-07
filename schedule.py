import streamlit as st

st.set_page_config(page_title="Smart Weekly Study Scheduler", page_icon="�")
st.title("📅 Smart Weekly Study Scheduler")

st.markdown("Enter your subjects and quiz scores to get a personalized weekly study plan!")

# Step 1: Get subjects and scores from user
num_topics = st.number_input("How many subjects do you want to schedule?", min_value=1, max_value=20, value=3)
topics = []
with st.form("subject_form"):
    for i in range(int(num_topics)):
        cols = st.columns([2,1])
        name = cols[0].text_input(f"Subject {i+1} name", key=f"name_{i}")
        score = cols[1].number_input(f"Score for {name or f'Subject {i+1}'} (0-100)", min_value=0, max_value=100, value=75, key=f"score_{i}")
        topics.append({"name": name, "score": score})
    submitted = st.form_submit_button("Generate Schedule")

# 🧠 Step 2: Assign priority and study duration
def assign_priority_and_duration(score):
    if score >= 80:
        return "Easy", 30
    elif score >= 60:
        return "Medium", 60
    else:
        return "Hard", 90


if submitted:
    week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    schedule = {day: [] for day in week_days}
    i = 0
    for topic in topics:
        if topic["name"]:
            priority, duration = assign_priority_and_duration(topic["score"])
            day = week_days[i % len(week_days)]
            schedule[day].append({
                "topic": topic["name"],
                "score": topic["score"],
                "priority": priority,
                "recommended_duration": f"{duration} min"
            })
            i += 1
    table_data = []
    for day, tasks in schedule.items():
        if tasks:
            for task in tasks:
                table_data.append([day, task['topic'], task['priority'], task['recommended_duration'], f"{task['score']}%"])
        else:
            table_data.append([day, "—", "—", "—", "—"])
    st.markdown("## � Weekly Smart Study Schedule")
    st.table(table_data)


