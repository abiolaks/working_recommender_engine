import pandas as pd
import time
from banking_recommender import BankingRecommendationSystem

import streamlit as st

# Set up Streamlit page configuration
st.set_page_config(
    page_title="BankAI Recommendations",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


class BankingUI:
    def __init__(self, recommender):
        self.recommender = recommender

    def _customer_profile_card(self, customer_data):
        """Display customer profile with formatted datetime"""
        with st.container(border=True):
            st.subheader("👤 Customer Profile")
            cols = st.columns(3)
            cols[0].metric(
                "Tenure", f"{customer_data['customer_tenure'].mean():.1f} years"
            )
            cols[1].metric(
                "Monthly Transactions",
                f"{customer_data['transaction_frequency'].mean():.1f}",
            )
            cols[2].metric("Favorite Product", customer_data["product_used"].mode()[0])

            st.write("📝 **Recent Transactions**")
            formatted_data = customer_data.copy()
            formatted_data["timestamp"] = formatted_data["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            st.dataframe(
                formatted_data[
                    [
                        "timestamp",
                        "transaction_type",
                        "transaction_amount",
                        "product_used",
                    ]
                ].head(10),
                hide_index=True,
                use_container_width=True,
            )

    def _recommendations_table(self, recommendations_df):
        """Display recommendations with proper product formatting"""
        with st.container(border=True):
            st.subheader("📊 Batch Recommendations")
            st.dataframe(
                recommendations_df,
                column_config={
                    "Recommended Products": st.column_config.TextColumn(
                        "Recommendations", width="medium"
                    ),
                    "Personalized Message": st.column_config.TextColumn(
                        "Message", width="large"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

    def show_main_interface(self):
        with st.sidebar:
            st.header("⚙️ Settings")
            uploaded_file = st.file_uploader(
                "Upload transaction data (CSV)/ API integration", type="csv"
            )
            self.recommender.load_data(uploaded_file)

            num_customers = st.slider(
                "Number of Customers to Process",
                1,
                100,
                10,
                help="Select how many customer recommendations to generate",
            )
            #st.subheader("Anti money laundering (AML Api integration)")
            #st.subheader("KYC (API integration)")
            
            st.markdown("""
            - 🛡️ Anti Money Laundering (AML API Integration)
            - 👤 KYC (API Integration)
            """)



        st.title("🏦 Smart Banking Recommendations")
        st.caption(
            "AI-powered product recommendations for personalized banking experiences"
        )
        if self.recommender.transaction_data is not None:
            # Display recommendations for selected customer
            customer_ids = self.recommender.transaction_data["customer_ID"].unique()
            selected_id = st.selectbox("Select Customer", customer_ids)
            selected_customer = self.recommender.transaction_data[
                self.recommender.transaction_data["customer_ID"] == selected_id
            ]

            self._customer_profile_card(selected_customer)

            # Display recommendations
            with st.container(border=True):
                st.subheader("🎯 Personalized Recommendations")
                recommendations = self.recommender.get_recommendations(selected_id)

                # ensure that we get recommendations
                if not recommendations:
                    st.warning(
                        "No specific recommendations found. Showing popular products instead."
                    )
                    recommendations = self.recommender._cold_start_recommendations(3)

                message = self.recommender.generate_message(
                    selected_customer, recommendations
                )

                cols = st.columns([1, 2])
                with cols[0]:
                    st.write("**Recommended Products**")
                    if recommendations:
                        for product in recommendations:
                            st.success(f"🌟 {product.replace('_', ' ').title()}")
                    else:
                        st.warning("No recommendations available")

                with cols[1]:
                    st.write("**AI-Powered Message**")
                    st.write(message)

            # Batch processing
            if st.button("🚀 Generate Batch Recommendations", use_container_width=True):
                with st.spinner("Generating recommendations..."):
                    start_time = time.time()
                    results = []
                    customer_ids = self.recommender.transaction_data[
                        "customer_ID"
                    ].unique()[:num_customers]

                    for cid in customer_ids:
                        customer_data = self.recommender.transaction_data[
                            self.recommender.transaction_data["customer_ID"] == cid
                        ]
                        recommendations = self.recommender.get_recommendations(cid)

                        message = self.recommender.generate_message(
                            customer_data, recommendations
                        )

                        results.append(
                            {
                                "Customer ID": cid,
                                "Recommended Products": "\n".join(
                                    [
                                        p.replace("_", " ").title()
                                        for p in recommendations
                                    ]
                                ),
                                "Personalized Message": message,
                                "Tenure": customer_data["customer_tenure"].mean(),
                                "Transaction Freq": customer_data[
                                    "transaction_frequency"
                                ].mean(),
                            }
                        )

                    self._recommendations_table(pd.DataFrame(results))
                    st.toast(
                        f"Generated {len(results)} recommendations in {time.time()-start_time:.2f}s",
                        icon="✅",
                    )
        else:
            st.write("No data uploaded. Please upload a CSV file.")


if __name__ == "__main__":
    # Use the get_azure_openai_client method for consistent Azure OpenAI client initialization
    recommender = BankingRecommendationSystem()
    recommender.openai_client = recommender.get_azure_openai_client()
    ui = BankingUI(recommender)
    ui.show_main_interface()
