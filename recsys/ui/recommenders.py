import time
from datetime import datetime

import streamlit as st
from sentence_transformers import SentenceTransformer

from recsys.config import settings

from .feature_group_updater import get_fg_updater
from .interaction_tracker import get_tracker
from .utils import (
    fetch_and_process_image,
    get_item_image_url,
    print_header,
    process_description,
)


def display_item(item_id, score, articles_fv, customer_id, tracker, source):
    """Display a single item with its interactions"""
    image_url = get_item_image_url(item_id, articles_fv)
    img = fetch_and_process_image(image_url)

    if img:
        st.image(img, use_column_width=True)
        st.write(f"**🎯 Score:** {score:.4f}")

        # View Details button
        details_key = f"{source}_details_{item_id}"
        if st.button("📝 View Details", key=details_key):
            tracker.track(customer_id, item_id, "click")
            with st.expander("Item Details", expanded=True):
                description = process_description(
                    articles_fv.get_feature_vector({"article_id": item_id})[-2]
                )
                st.write(description)

        # Buy button
        buy_key = f"{source}_buy_{item_id}"
        if st.button("🛒 Buy", key=buy_key):
            # Track interaction
            tracker.track(customer_id, item_id, "purchase")

            # Insert transaction
            fg_updater = get_fg_updater()
            purchase_data = {"customer_id": customer_id, "article_id": item_id}

            if fg_updater.insert_transaction(purchase_data):
                st.success(f"✅ Item {item_id} purchased!")
                st.experimental_rerun()
            else:
                st.error("Failed to record transaction, but purchase was tracked")


def customer_recommendations(
    articles_fv,
    ranking_deployment,
    query_model_deployment,
    customer_id,
    max_retries: int = 5,
    retry_delay: int = 30,
):
    """Handle customer-based recommendations"""
    tracker = get_tracker()

    # Initialize or update recommendations
    if "customer_recs" not in st.session_state:
        st.session_state.customer_recs = []
        st.session_state.prediction_time = None

    # Only get new predictions if:
    # 1. Button is clicked OR
    # 2. No recommendations exist OR
    # 3. Customer ID changed
    if (
        st.sidebar.button("Get Recommendations", key="get_recommendations_button")
        or not st.session_state.customer_recs
        or "last_customer_id" not in st.session_state
        or st.session_state.last_customer_id != customer_id
    ):
        with st.spinner("🔮 Getting recommendations..."):
            # Format timestamp with microseconds
            current_time = datetime.now()
            formatted_timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

            st.session_state.prediction_time = formatted_timestamp
            st.session_state.last_customer_id = customer_id

            # Get predictions from model using a retry mechanism in case of failure.
            deployment_input = [
                {"customer_id": customer_id, "transaction_date": formatted_timestamp}
            ]
            warning_placeholder = None
            for attempt in range(max_retries):
                try:
                    prediction = query_model_deployment.predict(
                        inputs=deployment_input
                    )["predictions"]["ranking"]
                    if warning_placeholder:
                        warning_placeholder.empty()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        warning_placeholder = st.warning(
                            f"⚠️ Failed to call the H&M recommender deployment. It's probably scaling from 0 to +1 instances, which may take 1-2 minutes. Retrying in {retry_delay} seconds..."
                        )
                        time.sleep(retry_delay)
                    else:
                        st.error(
                            f"❌ Failed to get predictions after {max_retries} retries"
                        )
                        raise e

            # Filter out purchased items
            available_items = [
                (item_id, score)
                for score, item_id in prediction
                if tracker.should_show_item(customer_id, item_id)
            ]

            # Store recommendations and extras
            st.session_state.customer_recs = available_items[:12]
            st.session_state.extra_recs = available_items[12:]

            # Track shown items
            tracker.track_shown_items(
                customer_id,
                [(item_id, score) for item_id, score in st.session_state.customer_recs],
            )

            st.sidebar.success("✅ Got new recommendations")

    # Display recommendations
    print_header("📝 Top 12 Recommendations:")

    if not st.session_state.customer_recs:
        st.warning(
            "No recommendations available. Click 'Get Recommendations' to start."
        )
        return

    # Display items in 3x4 grid
    for row in range(3):
        cols = st.columns(4)
        for col in range(4):
            idx = row * 4 + col
            if idx < len(st.session_state.customer_recs):
                item_id, score = st.session_state.customer_recs[idx]
                if tracker.should_show_item(customer_id, item_id):
                    with cols[col]:
                        display_item(
                            item_id,
                            score,
                            articles_fv,
                            customer_id,
                            tracker,
                            "customer",
                        )
                else:
                    # Replace purchased item with one from extras
                    if st.session_state.extra_recs:
                        new_item = st.session_state.extra_recs.pop(0)
                        st.session_state.customer_recs.append(new_item)
                    st.session_state.customer_recs.pop(idx)
                    st.experimental_rerun()




def display_category_items(emoji, category, items, articles_fv, customer_id, tracker):
    """Display items for a category and handle purchases"""
    st.markdown(f"## {emoji} {category}")

    if items:
        st.write(f"**Recommendation: {items[0][0]}**")

        # Calculate number of rows needed
        items_per_row = 5
        num_rows = (len(items) + items_per_row - 1) // items_per_row

        need_rerun = False
        remaining_items = []

        # Display items row by row
        for row in range(num_rows):
            start_idx = row * items_per_row
            end_idx = min(start_idx + items_per_row, len(items))
            row_items = items[start_idx:end_idx]

            cols = st.columns(items_per_row)

            for idx, item_data in enumerate(row_items):
                if tracker.should_show_item(customer_id, item_data[1][0]):
                    with cols[idx]:
                        if True:
                            remaining_items.append(item_data)

        st.markdown("---")
        return need_rerun, remaining_items
    return False, []



def get_similar_items(description, embedding_model, articles_fv):
    """Get similar items based on description embedding"""
    description_embedding = embedding_model.encode(description)

    return articles_fv.find_neighbors(description_embedding, k=25)
