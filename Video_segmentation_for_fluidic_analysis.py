import cv2  # video reading package
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Function to analyze flow and return flow values and cumulative flow
def analyze_flow(video_path, interval=500):
    cap = cv2.VideoCapture(video_path)

    flow_values = []
    frame_count = 0
    frame_numbers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([82, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Sum of blue pixels
        flow_value = np.sum(mask)
        flow_values.append(flow_value)
        frame_numbers.append(frame_count)  # Track frame numbers
        frame_count += 1

        # Displaying a frame each "interval" frame
        if frame_count % interval == 0:
            cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Original Frame', 500, 800)
            cv2.imshow('Original Frame', frame)
            cv2.namedWindow('HSV Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('HSV Frame', 500, 800)
            cv2.imshow('HSV Frame', cv2.bitwise_and(frame, frame, mask=mask))
            cv2.waitKey(0)  # clic to go display next frame

    cap.release()
    cv2.destroyAllWindows()

    # Calculate cumulative flow of blue pixels
    cumulative_flow = np.cumsum(flow_values)

    # Return flow values and cumulative flow
    return np.array(flow_values), cumulative_flow, np.array(frame_numbers)


# List of video paths and labels
video_paths = [
    ("C:/Users/Justine/Documents/UTC/TN09 - UTokyo/Photos_Video/100µLmn.MP4", "Video 1"),
    ("C:/Users/Justine/Documents/UTC/TN09 - UTokyo/Photos_Video/250µLmn.MP4", "Video 2"),
    ("C:/Users/Justine/Documents/UTC/TN09 - UTokyo/Photos_Video/500µLmn.MP4", "Video 3"),
    ("C:/Users/Justine/Documents/UTC/TN09 - UTokyo/Photos_Video/1000µLmn.MP4", "Video 4"),
]

# First Figure: Flux Over Time with Linear Regression
plt.figure(figsize=(12, 6))
for video_path, label in video_paths:
    flow_values, cumulative_flow, frame_numbers = analyze_flow(video_path)

    # Linear regression model
    frame_numbers_reshaped = frame_numbers.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(frame_numbers_reshaped, flow_values)
    predicted_flow = linear_model.predict(frame_numbers_reshaped)
    r_squared = linear_model.score(frame_numbers_reshaped, flow_values)

    # Plot flow values and regression line for each video
    plt.plot(frame_numbers, flow_values, label=f'{label} - Blue Pixels')
    plt.plot(frame_numbers, predicted_flow, linestyle='--', label=f'{label} - Regression (R²={r_squared:.2f})')

# Customize the plot for flow
plt.xlabel('Frames')
plt.ylabel('Number of Blue Pixels (Flow)')
plt.title('Flux Over Time with Linear Regression')
plt.legend()
plt.show()

# Second Figure: Cumulative Flow Curve
plt.figure(figsize=(12, 6))
for video_path, label in video_paths:
    _, cumulative_flow, _ = analyze_flow(video_path)

    # Plot cumulative flow curve
    plt.plot(cumulative_flow, label=f'{label} - Cumulative Flow')

# Customize the plot for cumulative flow
plt.xlabel('Frame Index')
plt.ylabel('Cumulative Blue Pixel Count')
plt.title('Cumulative Flow of Blue Pixels (Cumulative Count)')
plt.legend()
plt.grid()
plt.show()
