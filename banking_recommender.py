import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import os

BANKING_PRODUCTS = [
    {
        "name": "basic_checking_account",
        "description": "A no-frills checking account with low fees.",
        "category": "account",
    },
    {
        "name": "premium_checking_account",
        "description": "A high-tier checking account with benefits like cashback and no ATM fees.",
        "category": "account",
    },
    {
        "name": "savings_account",
        "description": "A standard savings account with competitive interest rates.",
        "category": "account",
    },
    {
        "name": "high_yield_savings_account",
        "description": "A savings account with higher interest rates for larger balances.",
        "category": "account",
    },
    {
        "name": "credit_card",
        "description": "A standard credit card with rewards on everyday purchases.",
        "category": "credit",
    },
    {
        "name": "platinum_credit_card",
        "description": "A premium credit card with travel rewards and concierge services.",
        "category": "credit",
    },
    {
        "name": "personal_loan",
        "description": "A loan for personal expenses with flexible repayment terms.",
        "category": "loan",
    },
    {
        "name": "low_interest_loan",
        "description": "A loan with lower interest rates for qualified customers.",
        "category": "loan",
    },
    {
        "name": "investment_account",
        "description": "An account for investing in stocks, bonds, and mutual funds.",
        "category": "investment",
    },
    {
        "name": "retirement_account",
        "description": "A tax-advantaged account for retirement savings.",
        "category": "investment",
    },
]


class BankingRecommendationSystem:
    def __init__(self, openai_api_key=None):
        if openai_api_key is None:
            # Try environment variable first, then Streamlit secrets
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if (
                not openai_api_key
                and hasattr(st, "secrets")
                and "OPENAI_API_KEY" in st.secrets
            ):
                openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.transaction_data = None
        self.customer_product_matrix = None
        self.openai_client = None

    def load_data(self, uploaded_file=None):
        """Load and preprocess transaction data"""
        if uploaded_file is not None:
            self.transaction_data = pd.read_csv(uploaded_file)
            self.transaction_data["timestamp"] = pd.to_datetime(
                self.transaction_data["timestamp"]
            )
            self._preprocess_data()
        else:
            st.write("No data uploaded. Please upload a CSV file.")

    def _preprocess_data(self):
        self.customer_product_matrix = self.transaction_data.pivot_table(
            index="customer_ID",
            columns="product_used",
            values="transaction_amount",
            aggfunc="count",
            fill_value=0,
        )

    def get_recommendations(self, customer_id, top_n=3):
        """Get recommendations for a customer with fallback logic"""
        if customer_id in self.customer_product_matrix.index:
            recommendations = self._collaborative_filtering(customer_id, top_n)
            # If collaborative filtering returns too few results, supplement with popular products
            if len(recommendations) < top_n:
                popular = self._cold_start_recommendations(top_n)
                recommendations.extend(p for p in popular if p not in recommendations)
                recommendations = recommendations[:top_n]
        else:
            recommendations = self._cold_start_recommendations(top_n)
        return recommendations

    def _collaborative_filtering(self, customer_id, top_n):
        """Improved collaborative filtering with fallback"""
        try:
            # Get products used by similar customers
            similarity_matrix = cosine_similarity(self.customer_product_matrix)
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=self.customer_product_matrix.index,
                columns=self.customer_product_matrix.index,
            )

            # Find similar customers (excluding self)
            similar_customers = (
                similarity_df[customer_id].sort_values(ascending=False).index[1:6]
            )  # Top 5 similar

            # Get products used by similar customers
            similar_products = (
                self.customer_product_matrix.loc[similar_customers]
                .sum()
                .sort_values(ascending=False)
            )

            # Filter out products already used by the customer
            used_products = set(
                self.customer_product_matrix.loc[customer_id][
                    self.customer_product_matrix.loc[customer_id] > 0
                ].index
            )

            # Get top N recommendations excluding used products
            recommendations = [
                product
                for product in similar_products.index
                if product not in used_products
            ][:top_n]

            return recommendations

        except Exception as e:
            print(f"Error in collaborative filtering: {str(e)}")
            return self._cold_start_recommendations(top_n)

    def _cold_start_recommendations(self, top_n):
        """Get popular products with fallback to all products"""
        try:
            popular_products = (
                self.transaction_data["product_used"].value_counts().index
            )
            return list(popular_products[:top_n])
        except Exception:
            # Fallback to all products if no transaction data
            return [product["name"] for product in BANKING_PRODUCTS][:top_n]

    def get_azure_openai_client(self):
        """Initialize and return the Azure OpenAI API client using st.secrets (local/Streamlit) or os.environ (Azure deployment)."""

        # Try st.secrets first, then fallback to os.environ
        api_key = st.secrets.get("AZURE_OPENAI_API_KEY") or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        azure_endpoint = st.secrets.get("ENDPOINT_URL") or os.environ.get(
            "ENDPOINT_URL"
        )
        deployment_name = st.secrets.get("DEPLOYMENT_NAME") or os.environ.get(
            "DEPLOYMENT_NAME"
        )
        api_version = (
            st.secrets.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or "2024-02-15-preview"
        )
        if not api_key or not azure_endpoint or not deployment_name:
            st.error(
                "Azure OpenAI configuration missing. Please set your API key, endpoint, and deployment name in .streamlit/secrets.toml or as environment variables."
            )
            return None
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
        )

    def generate_message(self, customer_data, recommended_products):
        """Generate personalized message with proper error handling using Azure OpenAI client from get_azure_openai_client."""
        openai_client = self.get_azure_openai_client()
        if not openai_client:
            return "Enable AI messaging by setting API key and Azure OpenAI config"

        try:
            prompt = f"""Generate a banking recommendation message for:
            - Tenure: {customer_data['customer_tenure'].mean():.1f} years
            - Transactions: {customer_data['transaction_frequency'].mean():.1f}/month
            - Favorite Product: {customer_data['product_used'].mode()[0]}
            Recommend: {', '.join(recommended_products)}
            """

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Message Error: {str(e)}"
