    if mode == "Automatic Detection":
        annotated_image, pill_count, detected_pills = detect_pills_pipeline(st.session_state.img, params)
        st.image(annotated_image, channels="BGR", caption=f"Found {pill_count} pill(s)")

        if detected_pills:
            st.write("---")
            st.subheader("Pill Summary")
            # Create a DataFrame for easy counting
            df = pd.DataFrame(detected_pills)
            summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
            st.table(summary_df)

    elif mode == "Manual ROI Matching":
        if st.button("Find Matching Pills"):
            # 1. Analyze the cropped ROI to identify the target pill
            cropped_img_cv = st.session_state.get('cropped_img')
            if cropped_img_cv is None:
                st.error("Please crop an image first.")
            else:
                # Use tight area parameters for the single pill in the ROI
                roi_params = {'min_area': 100, 'max_area': cropped_img_cv.shape[0] * cropped_img_cv.shape[1]}
                _, _, pills_in_roi = detect_pills_pipeline(cropped_img_cv, roi_params)

                if not pills_in_roi:
                    st.error("Could not detect a pill in the selected ROI. Try drawing a tighter box.")
                else:
                    target_pill = pills_in_roi[0] # Assume first pill is the target
                    target_shape = target_pill['shape']
                    target_color = target_pill['color']

                    # 2. Analyze the full image to find all pills
                    _, _, all_pills = detect_pills_pipeline(st.session_state.img, params)

                    # 3. Find matches
                    matches = []
                    for pill in all_pills:
                        if pill['shape'] == target_shape and pill['color'] == target_color:
                            matches.append(pill)

                    # 4. Draw rectangles on the original image for visualization
                    match_image = st.session_state.img.copy()
                    for pill in matches:
                        x, y, w, h = cv2.boundingRect(pill['contour'])
                        cv2.rectangle(match_image, (x, y), (x+w, y+h), (0, 255, 255), 4) # Yellow highlight

                    st.image(match_image, channels="BGR", caption=f"Highlighted {len(matches)} matching pill(s)")
                    st.write("---")
                    st.subheader("Matching Results")
                    match_data = {
                        'Shape': [target_shape],
                        'Color': [target_color],
                        'Quantity Found': [len(matches)]
                    }
                    st.table(pd.DataFrame(match_data))
