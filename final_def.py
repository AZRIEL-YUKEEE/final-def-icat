import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="K-Means Clustering", layout="wide")
st.title("Clustering Student ICAT Score Using Machine Learning Algorithm")

# ---- Fixed Course Mapping ----
course_mapping = {
    1: 'Bachelor of Secondary Education',
    2: 'Bachelor of Elementary Education',
    3: 'Bachelor of Physical Education',
    4: 'Bachelor of Science in Business Administration',
    5: 'Bachelor of Science in Mathematics',
    6: 'Bachelor of Arts in English Language',
    7: 'Bachelor of Arts in Psychology',
    8: 'Bachelor of Arts in Social Science',
    9: 'Bachelor of Science in Entrepreneurship',
    10: 'Bachelor of Science in Information Technology',
    11: 'Bachelor of Public Administration'
}

# ---- File Upload ----
uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # ---- Read file ----
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # ---- Clean column names ----
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # ---- Required columns ----
    required_cols = [
        "sex",
        "general_ability",
        "verbal_aptitude",
        "numerical_aptitude",
        "spatial_aptitude",
        "perceptual_aptitude",
        "manual_dexterity",
        "course"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        st.write("Your file contains:", list(df.columns))
        st.stop()

    # ---- Drop rows with missing values ----
    df = df.dropna(subset=required_cols)

    # ---- Convert Course to int ----
    df["course"] = df["course"].astype(int)

    # ---- Show Course Mapping Table ----
    st.subheader("📚 Course Code Mapping")
    st.table(pd.DataFrame(list(course_mapping.items()), columns=["Code", "Course"]))

    # ---- Insight Generator ----
    def generate_insight(x_col, y_col, avg_x, avg_y, overall_mean_x):
        if y_col == "sex":
            y_label = "Female" if avg_y < 1.5 else "Male"
            demographic_insight = f"This cluster is composed mostly of **{y_label}** students."
        elif y_col == "course":
            closest_code = min(course_mapping.keys(), key=lambda c: abs(c - avg_y))
            course_name = course_mapping[closest_code]
            demographic_insight = f"This cluster is composed mostly of students enrolled in **{course_name}**."
        else:
            demographic_insight = f"The average value for {y_col} is {avg_y:.2f}."

        if avg_x > overall_mean_x:
            ability_insight = f"The average **{x_col.replace('_',' ')}** score (**{avg_x:.2f}**) is **higher** than the overall mean."
        else:
            ability_insight = f"The average **{x_col.replace('_',' ')}** score (**{avg_x:.2f}**) is **lower** than the overall mean."

        overall = f"""
        - 📌 **Demographics:** {demographic_insight}
        - 🧮 **Performance Trend:** {ability_insight}
        - 🧠 **Interpretation:** This suggests that **{y_label if y_col=='sex' else course_name if y_col=='course' else y_col}** students 
          tend to have { 'stronger' if avg_x > overall_mean_x else 'weaker' } performance in 
          **{x_col.replace('_',' ')}** aptitude compared to other groups.
        """
        return overall

    # ---- Helper Function for Clustering and Plotting ----
    def cluster_and_plot(x_col, y_col, title, y_label):
        # ================================
        # 🔍 AUTO-DETECT BEST NUMBER OF K (Elbow Method)
        # ================================
        inertias = []
        K_range = range(2, 10)

        for k in K_range:
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_test.fit(df[[y_col, x_col]])
            inertias.append(kmeans_test.inertia_)

        # ---- Determine the "Elbow" point automatically ----
        knee = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
        best_k = knee.knee if knee.knee else 3  # Default to 3 if no clear elbow

        # ---- Run Final KMeans using the Best K ----
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(df[[y_col, x_col]])
        centers = kmeans.cluster_centers_

        # ---- Compute Evaluation Metrics ----
        silhouette_avg = silhouette_score(df[[y_col, x_col]], df["cluster"])
        ch_score = calinski_harabasz_score(df[[y_col, x_col]], df["cluster"])
        db_score = davies_bouldin_score(df[[y_col, x_col]], df["cluster"])

        # ---- Display Evaluation Metrics ----
        st.subheader("📊 Clustering Evaluation Metrics")
        st.markdown(f"**Optimal Number of Clusters (k):** {best_k}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{silhouette_avg:.3f}", "Higher is better")
        col2.metric("Calinski-Harabasz", f"{ch_score:.2f}", "Higher is better")
        col3.metric("Davies-Bouldin", f"{db_score:.3f}", "Lower is better")

        # ---- Elbow Plot ----
        with st.expander("📈 Elbow Method Visualization"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(K_range, inertias, 'bo-')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia (SSE)")
            ax.set_title("Elbow Method")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.info(f"✅ The **best k = {best_k}**, determined using the Elbow Method.")

        # ---- Scatter Plot ----
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            df[x_col], df[y_col],
            c=df["cluster"], cmap='viridis',
            s=80, edgecolors='w', linewidth=0.5
        )
        ax.scatter(
            centers[:, 1], centers[:, 0],
            c="red", s=200, marker="X", label="Cluster Centers"
        )
        ax.set_title(f"{title}\nBest K = {best_k} | Silhouette = {silhouette_avg:.3f}")
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_label)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # ---- Cluster Summary ----
        st.subheader("📋 Cluster Summary")
        summary = df.groupby("cluster")[[y_col, x_col]].mean().round(2)
        st.table(summary)

        # ---- Insight Section ----
        st.subheader("🧠 Automated Insight & Interpretation")
        overall_mean_x = df[x_col].mean()
        for cluster_id, row in summary.iterrows():
            avg_y = row[y_col]
            avg_x = row[x_col]
            insight = generate_insight(x_col, y_col, avg_x, avg_y, overall_mean_x)
            st.markdown(f"### 🟡 Cluster {cluster_id}\n{insight}")

        st.info(f"💬 The {title} visualization shows how students group by {y_label} and their {x_col.replace('_', ' ')} performance.")

    # ============================
    # 📊 SEX VS APTITUDE PLOTS
    # ============================
    st.header("📌 SEX vs Aptitude Scores")
    cluster_and_plot("general_ability", "sex", "K-Means Cluster Map (Sex vs General Ability)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("verbal_aptitude", "sex", "K-Means Cluster Map (Sex vs Verbal Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("numerical_aptitude", "sex", "K-Means Cluster Map (Sex vs Numerical Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("spatial_aptitude", "sex", "K-Means Cluster Map (Sex vs Spatial Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("perceptual_aptitude", "sex", "K-Means Cluster Map (Sex vs Perceptual Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("manual_dexterity", "sex", "K-Means Cluster Map (Sex vs Manual Dexterity)", "Sex (1=Female, 2=Male)")

    # ============================
    # 📚 COURSE VS APTITUDE PLOTS
    # ============================
    st.header("📌 COURSE vs Aptitude Scores")
    cluster_and_plot("general_ability", "course", "K-Means Cluster Map (Course vs General Ability)", "Course Code")
    cluster_and_plot("verbal_aptitude", "course", "K-Means Cluster Map (Course vs Verbal Aptitude)", "Course Code")
    cluster_and_plot("numerical_aptitude", "course", "K-Means Cluster Map (Course vs Numerical Aptitude)", "Course Code")
    cluster_and_plot("spatial_aptitude", "course", "K-Means Cluster Map (Course vs Spatial Aptitude)", "Course Code")
    cluster_and_plot("perceptual_aptitude", "course", "K-Means Cluster Map (Course vs Perceptual Aptitude)", "Course Code")
    cluster_and_plot("manual_dexterity", "course", "K-Means Cluster Map (Course vs Manual Dexterity)", "Course Code")

else:
    st.info("Please upload your CSV or Excel file to generate the cluster maps.")
