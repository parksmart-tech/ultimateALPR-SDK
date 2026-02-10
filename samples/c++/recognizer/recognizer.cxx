/* Copyright (C) 2011-2020 Doubango Telecom <https://www.doubango.org>
 * File author: Mamadou DIOP (Doubango Telecom, France).
 * License: For non commercial use only.
 * Source code: https://github.com/DoubangoTelecom/ultimateALPR-SDK
 * WebSite: https://www.doubango.org/webapps/alpr/
 */

/*
	https://github.com/DoubangoTelecom/ultimateALPR/blob/master/SDK_dist/samples/c++/recognizer/README.md
	Usage:
		recognizer \
			[--parallel <whether-to-enable-parallel-mode:true/false>] \
			[--rectify <whether-to-enable-rectification-layer:true/false>] \
			[--assets <path-to-assets-folder>] \
			[--charset <recognition-charset:latin/korean/chinese>] \
			[--car_noplate_detect_enabled <whether-to-enable-detecting-cars-with-no-plate:true/false>] \
			[--ienv_enabled <whether-to-enable-IENV:true/false>] \
			[--openvino_enabled <whether-to-enable-OpenVINO:true/false>] \
			[--openvino_device <openvino_device-to-use>] \
			[--npu_enabled <whether-to-enable-NPU-acceleration:true/false>] \
			[--trt_enabled <whether-to-enable-TensorRT-acceleration:true/false>] \
			[--klass_lpci_enabled <whether-to-enable-LPCI:true/false>] \
			[--klass_vcr_enabled <whether-to-enable-VCR:true/false>] \
			[--klass_vmmr_enabled <whether-to-enable-VMMR:true/false>] \
			[--klass_vbsr_enabled <whether-to-enable-VBSR:true/false>] \
			[--tokenfile <path-to-license-token-file>] \
			[--tokendata <base64-license-token-data>]

	Example:
		recognizer \
			--image C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dist/assets/images/lic_us_1280x720.jpg \
			--parallel true \
			--rectify false \
			--assets C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dist/assets \
			--charset latin \
			--tokenfile C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dev/tokens/windows-iMac.lic

*/

#include <ultimateALPR-SDK-API-PUBLIC.h>

#include <stdio.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <regex>
#include <memory>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
using namespace rapidjson;

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/core/utils/logger.hpp>
using namespace cv;

#ifdef RPI
#include <pigpio.h>
#endif
#include <arpa/inet.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream> // std::cout
#include <sstream>
#include <sys/stat.h>
#include <ctime>
#include <cstdint>
#include <map>
#if defined(_WIN32)
#include <Windows.h> // SetConsoleOutputCP
#include <algorithm> // std::replace
#endif

// Not part of the SDK, used to decode images -> https://github.com/nothings/stb
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "../stb_image.h"

using namespace ultimateAlprSdk;

// Asset manager used on Android to files in "assets" folder
#if ULTALPR_SDK_OS_ANDROID
#define ASSET_MGR_PARAM() __sdk_android_assetmgr,
#else
#define ASSET_MGR_PARAM()
#endif /* ULTALPR_SDK_OS_ANDROID */

cv::Mat frames[4][100];

struct Vehicle
{
	cv::Rect bounding_box;
	cv::Point centroid;
	cv::Rect vehicle_bounding_box;
	cv::Point vehicle_centroid;
	std::string vehicle_number;
	std::string valid_vehicle_number;
	float confidence;
	float valid_confidence;
	int tracking_count;
	std::string message_id;
	std::chrono::time_point<std::chrono::system_clock> first_seen_at;
	std::chrono::time_point<std::chrono::system_clock> last_seen_at;
	std::chrono::time_point<std::chrono::system_clock> published_at;
	bool published;
	cv::Mat captured_frame;
	cv::Mat centre_frame;
	cv::Mat valid_frame;
	cv::Mat valid_centre_frame;
	cv::Point initial_position;
	cv::Point final_position;
	bool hidden_number_plate;
	bool passed_through_centre;
	std::string body_style;
	float body_style_confidence;
};

struct FrameItem {
    std::shared_ptr<cv::Mat> frame;
    std::chrono::steady_clock::time_point timestamp;
};

class FrameQueue 
{
public:
	FrameQueue(size_t max_size) : max_size_(max_size) {}

	void push(std::shared_ptr<cv::Mat> frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop_front();
        }
        queue_.push_back(
			{
				frame, 
				std::chrono::steady_clock::now()
			}
		);
    }

	std::shared_ptr<cv::Mat> popRecentOnly(int max_age_ms = 1000) {
		std::lock_guard<std::mutex> lock(mutex_);
	
		auto now = std::chrono::steady_clock::now();
	
		while (!queue_.empty()) {
			auto& item = queue_.front();
			auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - item.timestamp).count();
	
			if (age_ms <= max_age_ms) {
				auto frame = item.frame;
				queue_.pop_front();
				return frame;
			}
	
			// Drop stale frame
			queue_.pop_front();
		}
	
		return nullptr; // No fresh frame found
	}
	
	size_t size() {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.size();
	}
	
private:
	std::deque<FrameItem> queue_;
	std::mutex mutex_;
	size_t max_size_;
};

static int64_t nowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
			   std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

static void configure_capture(const std::shared_ptr<cv::VideoCapture> &cap_ptr, int index)
{
#ifdef CAP_PROP_BUFFERSIZE
	if (!cap_ptr->set(CAP_PROP_BUFFERSIZE, 1))
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d Failed to set CAP_PROP_BUFFERSIZE", index);
	}
#endif
#ifdef CAP_PROP_OPEN_TIMEOUT_MSEC
	if (!cap_ptr->set(CAP_PROP_OPEN_TIMEOUT_MSEC, 5000))
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d Failed to set CAP_PROP_OPEN_TIMEOUT_MSEC", index);
	}
#endif
#ifdef CAP_PROP_READ_TIMEOUT_MSEC
	if (!cap_ptr->set(CAP_PROP_READ_TIMEOUT_MSEC, 5000))
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d Failed to set CAP_PROP_READ_TIMEOUT_MSEC", index);
	}
#endif
}

static void printUsage(const std::string &message = "");
static bool parseArgs(int argc, char *argv[], std::map<std::string, std::string> &values);
static void latest_frames(const std::string rtsp_path, bool usePipeline, std::string encoding, std::atomic<float> &fs, std::atomic<int> &frame_width, std::atomic<int> &frame_height, std::shared_ptr<FrameQueue> frame_queue_ptr, std::shared_ptr<cv::VideoCapture> cap_ptr, std::atomic<bool> &end_thread, std::atomic<double> &start_time, std::atomic<bool> &stream_started, std::atomic<int64_t> &last_frame_ms, int index);
#ifdef RPI
static void process_image(Mat frame, bool isParallelDeliveryEnabled, int index, const std::string server_url, int port, int relay, float &f, float *FPS, int &frame_count, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, float minimumConfidence, bool determineDirection, bool trackingEnabled, std::vector<Vehicle> &vehicles, int* fps_counter, bool isGpioInitialised, int gpio_pin, int default_gpio_level);
#else
static void process_image(Mat frame, bool isParallelDeliveryEnabled, int index, const std::string server_url, int port, int relay, float &f, float *FPS, int &frame_count, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, float minimumConfidence, bool determineDirection, bool trackingEnabled, std::vector<Vehicle> &vehicles, int* fps_counter);
#endif
static int send_message(const std::string server_url, const int port, const std::string message, int index);
static std::string print_random_string(int n);
#ifdef RPI
static void process_rtsp_stream(const std::string rtsp_url, int index, bool isParallelDeliveryEnabled, const std::string server_url, int port, int relay, std::string windowNamePrefix, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, bool usePipeline, float minimumConfidence, bool determineDirection, std::string encoding, bool trackingEnabled, std::vector<Vehicle> (&vehicles)[4], bool isGpioInitialised, int gpio_pin, int default_gpio_level);
#else
static void process_rtsp_stream(const std::string rtsp_url, int index, bool isParallelDeliveryEnabled, const std::string server_url, int port, int relay, std::string windowNamePrefix, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, bool usePipeline, float minimumConfidence, bool determineDirection, std::string encoding, bool trackingEnabled, std::vector<Vehicle> (&vehicles)[4]);
#endif
static int levenshteinDistance(const std::string &s1, const std::string &s2);
static void process_image_result(UltAlprSdkResult result, Mat frame, int index, const std::string server_url, int port, int relay, std::string logtype, std::string description, std::string ftp, float minimumConfidence, bool determineDirection, bool trackingEnabled, std::vector<Vehicle> &vehicles, bool isShowFrame, bool pushRawData);
static int64_t nowMs();
static void configure_capture(const std::shared_ptr<cv::VideoCapture> &cap_ptr, int index);

// Configuration for ANPR deep learning engine
static const char *__jsonConfig =
	"{"
	"\"debug_level\": \"warn\","
	"\"debug_write_input_image_enabled\": false,"
	"\"debug_internal_data_path\": \".\","
	""
#ifdef RPI
	"\"num_threads\": 1,"
	"\"max_jobs\": 1,"
#else
	"\"num_threads\": -1,"
	"\"max_jobs\": -1,"
#endif
	"\"gpgpu_enabled\": true,"
	"\"asm_enabled\": true,"
	"\"intrin_enabled\": true,"
	""
	"\"klass_vcr_gamma\": 1.5,"
	""
	"\"detect_roi\": [0, 0, 0, 0],"
	"\"detect_minscore\": 0.1,"
	""
	"\"car_noplate_detect_min_score\": 0.8,"
	""
	"\"pyramidal_search_enabled\": true,"
	"\"pyramidal_search_sensitivity\": 1.0,"
	"\"pyramidal_search_minscore\": 0.3,"
	"\"pyramidal_search_min_image_size_inpixels\": 800,"
	""
	"\"recogn_minscore\": 0.3,"
	"\"recogn_score_type\": \"min\""
	"";

std::array<std::mutex, 4> frame_mtx;
std::unordered_map<int, int> frame_id_index_map;

class MyUltAlprSdkParallelDeliveryCallback : public UltAlprSdkParallelDeliveryCallback
{
public:
	MyUltAlprSdkParallelDeliveryCallback(const std::string &charset, std::string ftp, std::vector<std::string> logtypes, std::vector<std::string> descriptions, std::vector<std::string> server_urls, std::vector<int> ports, std::vector<int> relays, float minimumConfidence, bool determineDirection, bool pushRawData, bool trackingEnabled, std::vector<Vehicle> (&vehicles)[4]) : m_strCharset(charset), ftp(ftp), logtypes(logtypes), descriptions(descriptions), server_urls(server_urls), ports(ports), relays(relays), minimumConfidence(minimumConfidence), determineDirection(determineDirection), pushRawData(pushRawData), trackingEnabled(trackingEnabled), vehicles(vehicles) {}

	virtual void onNewResult(const UltAlprSdkResult *result) const override
	{
		// ULTALPR_SDK_ASSERT(result != nullptr);
		const std::string &json_ = result->json();
		if (!json_.empty())
		{
			Document document;
			document.Parse(json_.c_str());
			if (document.HasMember("frame_id"))
			{
				int frame_id = document["frame_id"].GetInt();
				frame_id = frame_id % 100;
				int index = frame_id_index_map[frame_id];
				cv::Mat frame = frames[index][frame_id];
				cv::Mat original_frame = frame.clone();
				UltAlprSdkResult alprResult = *result;
				process_image_result(alprResult, frame, index, server_urls.at(index), ports.at(index), relays.at(index), logtypes.at(index), descriptions.at(index), ftp, minimumConfidence, determineDirection, trackingEnabled, vehicles[index], false, pushRawData);
			}
		}
	}

private:
	std::string ftp;
	std::vector<std::string> logtypes;
	std::vector<std::string> descriptions;
	std::vector<std::string> server_urls;
	std::vector<int> ports;
	std::vector<int> relays;
	float minimumConfidence;
	bool determineDirection;
	bool pushRawData;
	std::string m_strCharset;
	bool trackingEnabled;
	std::vector<Vehicle> (&vehicles)[4];
};

/*
 * Entry point
 */
int main(int argc, char *argv[])
{
	srand(time(NULL));
	// Activate UT8 display
#if defined(_WIN32)
	SetConsoleOutputCP(CP_UTF8);
#endif

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

#if defined(JETSON) || defined(ORIN)
	int device_id = 0;
	cv::cuda::setDevice(device_id);
#endif

	// local variables
	UltAlprSdkResult result;
	std::string assetsFolder, licenseTokenData, licenseTokenFile;
	bool isParallelDeliveryEnabled = false; // Single image -> no need for parallel processing
	bool isRectificationEnabled = true;
	bool isCarNoPlateDetectEnabled = false;
	bool isIENVEnabled = false;
	bool isShowFrame = false;
	bool usePipeline = false;
	bool trackVehicles = false;
	bool isOpenVinoEnabled =
#if defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(_ARM) || defined(_M_ARM) || defined(_M_ARMT) || defined(__arm) || defined(__aarch64__)
		false;
#else // x86-64
		true;
#endif
	bool isNpuEnabled = true;	   // Amlogic, NXP...
	bool isTensorRTEnabled = true; // NVIDIA TensorRT
	bool isKlassLPCI_Enabled = false;
	bool isKlassVCR_Enabled = false;
	bool isKlassVMMR_Enabled = false;
	bool isKlassVBSR_Enabled = false;
#ifdef RPI
	bool isGpioInitialised = false;
	int gpio_pin = -1;
#endif
	std::string charset = "latin";
	std::string openvinoDevice = "CPU";
	std::string rtsp = "";
	std::string ftp = "";
	std::string server = "";
	std::string port = "";
	std::string relay = "";
	std::string logtype = "";
	std::string description = "";
	std::string windowNamePrefix = "";
#ifdef RPI
	std::string gpioString = "";
	int default_gpio_level = 1;
#endif
	std::string encoding = "H264";
	float minimumConfidence = 40.0f;
	bool determineDirection = false;
	bool pushRawData = false;
	// Parsing args
	std::map<std::string, std::string> args;
	if (!parseArgs(argc, argv, args))
	{
		printUsage();
		return -1;
	}

	if (args.find("--parallel") != args.end())
	{
		isParallelDeliveryEnabled = (args["--parallel"].compare("true") == 0);
	}
	if (args.find("--assets") != args.end())
	{
		assetsFolder = args["--assets"];
#if defined(_WIN32)
		std::replace(assetsFolder.begin(), assetsFolder.end(), '\\', '/');
#endif
	}
	if (args.find("--charset") != args.end())
	{
		charset = args["--charset"];
	}
	if (args.find("--rectify") != args.end())
	{
		isRectificationEnabled = (args["--rectify"].compare("true") == 0);
	}
	if (args.find("--showframe") != args.end())
	{
		isShowFrame = (args["--showframe"].compare("true") == 0);
	}
	if (args.find("--direction") != args.end())
	{
		determineDirection = (args["--direction"].compare("true") == 0);
	}
	if (args.find("--raw") != args.end())
	{
		pushRawData = (args["--raw"].compare("true") == 0);
	}
	if (args.find("--pipeline") != args.end())
	{
		usePipeline = (args["--pipeline"].compare("true") == 0);
	}
	if (args.find("--tracking") != args.end())
	{
		trackVehicles = (args["--tracking"].compare("true") == 0);
	}
	if (args.find("--car_noplate_detect_enabled") != args.end())
	{
		isCarNoPlateDetectEnabled = (args["--car_noplate_detect_enabled"].compare("true") == 0);
	}
	if (args.find("--ienv_enabled") != args.end())
	{
		isIENVEnabled = (args["--ienv_enabled"].compare("true") == 0);
	}
	if (args.find("--openvino_enabled") != args.end())
	{
		isOpenVinoEnabled = (args["--openvino_enabled"].compare("true") == 0);
	}
	if (args.find("--openvino_device") != args.end())
	{
		openvinoDevice = args["--openvino_device"];
	}
	if (args.find("--npu_enabled") != args.end())
	{
		isNpuEnabled = (args["--npu_enabled"].compare("true") == 0);
	}
	if (args.find("--trt_enabled") != args.end())
	{
		isTensorRTEnabled = (args["--trt_enabled"].compare("true") == 0);
	}
	if (args.find("--klass_lpci_enabled") != args.end())
	{
		isKlassLPCI_Enabled = (args["--klass_lpci_enabled"].compare("true") == 0);
	}
	if (args.find("--klass_vcr_enabled") != args.end())
	{
		isKlassVCR_Enabled = (args["--klass_vcr_enabled"].compare("true") == 0);
	}
	if (args.find("--klass_vmmr_enabled") != args.end())
	{
		isKlassVMMR_Enabled = (args["--klass_vmmr_enabled"].compare("true") == 0);
	}
	if (args.find("--klass_vbsr_enabled") != args.end())
	{
		isKlassVBSR_Enabled = (args["--klass_vbsr_enabled"].compare("true") == 0);
	}
	if (args.find("--tokenfile") != args.end())
	{
		licenseTokenFile = args["--tokenfile"];
#if defined(_WIN32)
		std::replace(licenseTokenFile.begin(), licenseTokenFile.end(), '\\', '/');
#endif
	}
	if (args.find("--tokendata") != args.end())
	{
		licenseTokenData = args["--tokendata"];
	}
	if (args.find("--encoding") != args.end())
	{
		encoding = args["--encoding"];
	}
	if (args.find("--rtsp") != args.end())
	{
		rtsp = args["--rtsp"];
	}
	if (args.find("--server") != args.end())
	{
		server = args["--server"];
	}
	if (args.find("--ftp") != args.end())
	{
		ftp = args["--ftp"];
	}
	if (args.find("--port") != args.end())
	{
		port = args["--port"];
	}
	if (args.find("--relay") != args.end())
	{
		relay = args["--relay"];
	}
#ifdef RPI
	if (args.find("--gpio") != args.end())
	{
		gpioString = args["--gpio"];
	}
	if (args.find("--default_gpio_level") != args.end())
	{
		if (args["--default_gpio_level"].compare("0") == 0)
		{
			default_gpio_level = 0;
		}
		else
		{
			default_gpio_level = 1;
		}
	}
#endif
	if (args.find("--logtype") != args.end())
	{
		logtype = args["--logtype"];
	}
	if (args.find("--min") != args.end())
	{
		minimumConfidence = std::stof(args["--min"]);
	}
	if (args.find("--description") != args.end())
	{
		description = args["--description"];
	}
	if (args.find("--prefix") != args.end())
	{
		windowNamePrefix = args["--prefix"];
	}
	if (rtsp == "")
	{
		ULTALPR_SDK_PRINT_INFO("RTSP is required");
		return -1;
	}
	if (server == "")
	{
		ULTALPR_SDK_PRINT_INFO("Server is required");
		return -1;
	}
	if (port == "")
	{
		ULTALPR_SDK_PRINT_INFO("Port is required");
		return -1;
	}
	if (relay == "")
	{
		ULTALPR_SDK_PRINT_INFO("Relay is required");
		return -1;
	}
	if (logtype == "")
	{
		ULTALPR_SDK_PRINT_INFO("Log type is required");
		return -1;
	}
	if (description == "")
	{
		ULTALPR_SDK_PRINT_INFO("Description is required");
		return -1;
	}
#ifdef RPI
	if (gpioString != "")
	{
		ULTALPR_SDK_PRINT_INFO("Initialising GPIO...");
		if (gpioInitialise() < 0)
		{
			ULTALPR_SDK_PRINT_INFO("Initialising GPIO failed");
		}
		else
		{
			gpio_pin = std::stoi(gpioString);
			gpioSetMode(gpio_pin, PI_INPUT);
			isGpioInitialised = true;
			ULTALPR_SDK_PRINT_INFO("Initialising GPIO successfull");
		}
	}
#endif

	// Update JSON config
	std::string jsonConfig = __jsonConfig;
	if (!assetsFolder.empty())
	{
		jsonConfig += std::string(",\"assets_folder\": \"") + assetsFolder + std::string("\"");
	}
	if (!charset.empty())
	{
		jsonConfig += std::string(",\"charset\": \"") + charset + std::string("\"");
	}
	jsonConfig += std::string(",\"recogn_rectify_enabled\": ") + (isRectificationEnabled ? "true" : "false");
	jsonConfig += std::string(",\"car_noplate_detect_enabled\": ") + (isCarNoPlateDetectEnabled ? "true" : "false");
	jsonConfig += std::string(",\"ienv_enabled\": ") + (isIENVEnabled ? "true" : "false");
	jsonConfig += std::string(",\"openvino_enabled\": ") + (isOpenVinoEnabled ? "true" : "false");
	if (!openvinoDevice.empty())
	{
		jsonConfig += std::string(",\"openvino_device\": \"") + openvinoDevice + std::string("\"");
	}
	jsonConfig += std::string(",\"npu_enabled\": ") + (isNpuEnabled ? "true" : "false");
	jsonConfig += std::string(",\"trt_enabled\": ") + (isTensorRTEnabled ? "true" : "false");
	jsonConfig += std::string(",\"klass_lpci_enabled\": ") + (isKlassLPCI_Enabled ? "true" : "false");
	jsonConfig += std::string(",\"klass_vcr_enabled\": ") + (isKlassVCR_Enabled ? "true" : "false");
	jsonConfig += std::string(",\"klass_vmmr_enabled\": ") + (isKlassVMMR_Enabled ? "true" : "false");
	jsonConfig += std::string(",\"klass_vbsr_enabled\": ") + (isKlassVBSR_Enabled ? "true" : "false");
	if (!licenseTokenFile.empty())
	{
		jsonConfig += std::string(",\"license_token_file\": \"") + licenseTokenFile + std::string("\"");
	}
	if (!licenseTokenData.empty())
	{
		jsonConfig += std::string(",\"license_token_data\": \"") + licenseTokenData + std::string("\"");
	}

	jsonConfig += "}"; // end-of-config

	std::vector<std::string> rtsp_urls;
	if (rtsp != "")
	{
		std::stringstream rtsp_stream(rtsp);
		while (rtsp_stream.good())
		{
			std::string substr;
			getline(rtsp_stream, substr, ',');
			rtsp_urls.push_back(substr);
		}
	}

	std::vector<std::string> server_urls;
	if (server != "")
	{
		std::stringstream server_stream(server);
		while (server_stream.good())
		{
			std::string substr;
			getline(server_stream, substr, ',');
			server_urls.push_back(substr);
		}
	}

	std::vector<int> ports;
	if (port != "")
	{
		std::stringstream port_stream(port);
		while (port_stream.good())
		{
			std::string substr;
			getline(port_stream, substr, ',');
			ports.push_back(std::stoi(substr));
		}
	}

	std::vector<int> relays;
	if (relay != "")
	{
		std::stringstream relay_stream(relay);
		while (relay_stream.good())
		{
			std::string substr;
			getline(relay_stream, substr, ',');
			relays.push_back(std::stoi(substr));
		}
	}

	std::vector<std::string> logtypes;

	if (logtype != "")
	{
		std::stringstream logtype_stream(logtype);
		while (logtype_stream.good())
		{
			std::string substr;
			getline(logtype_stream, substr, ',');
			logtypes.push_back(substr);
		}
	}

	std::vector<std::string> descriptions;
	if (description != "")
	{
		std::stringstream description_stream(description);
		while (description_stream.good())
		{
			std::string substr;
			getline(description_stream, substr, ',');
			descriptions.push_back(substr);
		}
	}

	if (rtsp_urls.size() != server_urls.size())
	{
		ULTALPR_SDK_PRINT_INFO("RTSP and Server URLs mismatch");
		return -1;
	}

	if (ports.size() != server_urls.size())
	{
		ULTALPR_SDK_PRINT_INFO("Ports and Server URLs mismatch");
		return -1;
	}

	if (relays.size() != server_urls.size())
	{
		ULTALPR_SDK_PRINT_INFO("Relays and Server URLs mismatch");
		return -1;
	}

	if (logtypes.size() != server_urls.size())
	{
		ULTALPR_SDK_PRINT_INFO("Log types and Server URLs mismatch");
		return -1;
	}

	if (descriptions.size() != server_urls.size())
	{
		ULTALPR_SDK_PRINT_INFO("Descriptions and Server URLs mismatch");
		return -1;
	}

	if (encoding != "H265" && encoding != "H264")
	{
		ULTALPR_SDK_PRINT_INFO("Invalid stream encoding");
	}

	// Init
	ULTALPR_SDK_PRINT_INFO("Starting recognizer...");

	std::vector<Vehicle> vehicles[4];

	MyUltAlprSdkParallelDeliveryCallback parallelDeliveryCallbackCallback(charset, ftp, logtypes, descriptions, server_urls, ports, relays, minimumConfidence, determineDirection, pushRawData, trackVehicles, vehicles);
	ULTALPR_SDK_ASSERT((result = UltAlprSdkEngine::init(
							ASSET_MGR_PARAM()
								jsonConfig.c_str(),
							isParallelDeliveryEnabled ? &parallelDeliveryCallbackCallback : nullptr))
						   .isOK());

	std::string pathFilePositive;

	ULTALPR_SDK_ASSERT((result = UltAlprSdkEngine::warmUp(
							ULTALPR_SDK_IMAGE_TYPE_RGB24))
						   .isOK());

#ifdef RPI
	ULTALPR_SDK_PRINT_INFO("Running on Raspberry Pi");
#elif defined(JETSON)
	ULTALPR_SDK_PRINT_INFO("Running on Jetson");
#elif defined(ORIN)
	ULTALPR_SDK_PRINT_INFO("Running on Orin");
#else
	ULTALPR_SDK_PRINT_INFO("Running on Uknown platform");
#endif

	ULTALPR_SDK_PRINT_INFO("Minimum confidence : %f", minimumConfidence);
	ULTALPR_SDK_PRINT_INFO("Determine direction : %d", determineDirection);

	std::vector<std::thread> rtsp_threads;
	for (int i = 0; i < rtsp_urls.size(); i++)
	{
		const std::string rtsp_path = rtsp_urls.at(i);
#ifdef RPI
		auto rtsp_thread = std::thread(process_rtsp_stream, rtsp_path, i, isParallelDeliveryEnabled, server_urls.at(i), ports.at(i), relays.at(i), windowNamePrefix, logtypes.at(i), descriptions.at(i), isShowFrame, pushRawData, ftp, usePipeline, minimumConfidence, determineDirection, encoding, trackVehicles, std::ref(vehicles), isGpioInitialised, gpio_pin, default_gpio_level);
#else
		auto rtsp_thread = std::thread(process_rtsp_stream, rtsp_path, i, isParallelDeliveryEnabled, server_urls.at(i), ports.at(i), relays.at(i), windowNamePrefix, logtypes.at(i), descriptions.at(i), isShowFrame, pushRawData, ftp, usePipeline, minimumConfidence, determineDirection, encoding, trackVehicles, std::ref(vehicles));
#endif
		rtsp_threads.push_back(move(rtsp_thread));
	}

	for (int i = 0; i < rtsp_threads.size(); i++)
	{
		rtsp_threads.at(i).join();
	}

	// DeInit
	ULTALPR_SDK_PRINT_INFO("Ending recognizer...");
	ULTALPR_SDK_ASSERT((result = UltAlprSdkEngine::deInit()).isOK());

	return 0;
}


#ifdef RPI
static void process_rtsp_stream(const std::string rtsp_path, int index, bool isParallelDeliveryEnabled, const std::string server_url, int port, int relay, std::string windowNamePrefix, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, bool usePipeline, float minimumConfidence, bool determineDirection, std::string encoding, bool trackingEnabled, std::vector<Vehicle> (&vehicles)[4], bool isGpioInitialised, int gpio_pin, int default_gpio_level)
#else
static void process_rtsp_stream(const std::string rtsp_path, int index, bool isParallelDeliveryEnabled, const std::string server_url, int port, int relay, std::string windowNamePrefix, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, bool usePipeline, float minimumConfidence, bool determineDirection, std::string encoding, bool trackingEnabled, std::vector<Vehicle> (&vehicles)[4])
#endif
{
	int fps_counter = 0;
	while (true)
	{
		try
		{
			const int64_t kNoFrameTimeoutMs = 5000;
			float f;
			std::atomic<int> frame_width(0.0);
			std::atomic<int> frame_height(0.0);
			std::atomic<float> fs(0.0);
			std::atomic<int64_t> last_frame_ms(0);
			float FPS[16];
			int frame_count = 0;
			for (int i = 0; i < 16; i++)
				FPS[i] = 0.0;

			ULTALPR_SDK_PRINT_INFO("Stream #%d Trying to watch Camera feed", index);
			ULTALPR_SDK_PRINT_INFO("Stream #%d RTSP URL : %s", index, rtsp_path.c_str());
			ULTALPR_SDK_PRINT_INFO("Stream #%d Server URL : %s:%d at Relay %d", index, server_url.c_str(), port, relay);

			std::string window_name = "anpr";
			window_name += std::to_string(index);

			if (isShowFrame)
			{
				namedWindow(window_name.c_str(), WINDOW_AUTOSIZE);
			}
			
			std::shared_ptr<FrameQueue> frame_queue_ptr = std::make_shared<FrameQueue>(30);
			std::shared_ptr<cv::VideoCapture> cap_ptr = std::make_shared<cv::VideoCapture>();

			std::atomic<bool> end_thread(false);
			std::atomic<bool> stream_started(false);
			std::atomic<double> start_time(0.0);
			ULTALPR_SDK_PRINT_INFO("Stream #%d Starting frames thread", index);
			auto frames_thread = std::thread(latest_frames, rtsp_path, usePipeline, encoding, std::ref(fs), std::ref(frame_width), std::ref(frame_height), frame_queue_ptr, cap_ptr, std::ref(end_thread), std::ref(start_time), std::ref(stream_started), std::ref(last_frame_ms), index);
			
			ULTALPR_SDK_PRINT_INFO("Stream #%d Analyzing frames", index);
			bool thread_handled = 0;
			while (!end_thread.load(std::memory_order_acquire))
			{

				
				if (!stream_started.load(std::memory_order_acquire))
				{
					ULTALPR_SDK_PRINT_INFO("Stream #%d Waiting for stream to start", index);
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				}
				

				double time_elapsed = (cv::getTickCount() - start_time.load(std::memory_order_acquire)) / cv::getTickFrequency();
				if (usePipeline)
				{
					if (!stream_started.load(std::memory_order_acquire))
					{
						ULTALPR_SDK_PRINT_INFO("Stream #%d Waiting for stream to start", index);
						std::this_thread::sleep_for(std::chrono::milliseconds(5000));
						continue;
					}
				}
				cv::Mat new_frame;
				while (!end_thread.load(std::memory_order_acquire))
				{
					int64_t last_ms = last_frame_ms.load(std::memory_order_acquire);
					if (stream_started.load(std::memory_order_acquire) && last_ms > 0)
					{
						int64_t now_ms = nowMs();
						if ((now_ms - last_ms) > kNoFrameTimeoutMs)
						{
							ULTALPR_SDK_PRINT_INFO("Stream #%d No frames for %lld ms. Restarting stream.", index, static_cast<long long>(now_ms - last_ms));
							end_thread.store(true, std::memory_order_release);
							if (cap_ptr->isOpened())
							{
								cap_ptr->release();
							}
							break;
						}
					}
					frame_mtx[index].lock();
					auto frame = frame_queue_ptr->popRecentOnly(300);
					frame_mtx[index].unlock();
					if (frame){
						new_frame = frame->clone();
						break;
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
				}
				
				if (new_frame.empty())
				{
					ULTALPR_SDK_PRINT_INFO("Stream #%d Empty frame in feed", index);
					end_thread.store(true, std::memory_order_release);
					frames_thread.join();
					thread_handled = 1;
					ULTALPR_SDK_PRINT_INFO("Stream #%d Thread joined", index);
					break;
				}
				else
				{
#ifdef RPI
					process_image(new_frame, isParallelDeliveryEnabled, index, server_url, port, relay, f, FPS, frame_count, logtype, description, isShowFrame, pushRawData, ftp, minimumConfidence, determineDirection, trackingEnabled, vehicles[index], &fps_counter, isGpioInitialised, gpio_pin, default_gpio_level);
#else
					process_image(new_frame, isParallelDeliveryEnabled, index, server_url, port, relay, f, FPS, frame_count, logtype, description, isShowFrame, pushRawData, ftp, minimumConfidence, determineDirection, trackingEnabled, vehicles[index], &fps_counter);
#endif
				}
			}
			if (!thread_handled)
			{
				frames_thread.join();
			}
			ULTALPR_SDK_PRINT_INFO("Stream #%d Stopped reading frame from RTSP Stream", index);
			cv::destroyWindow(window_name);
		}
		catch (const cv::Exception &e)
		{
			ULTALPR_SDK_PRINT_INFO("Stream #%d OpenCV exception caught: %s", index, e.what());
		}
		catch (const std::exception &e)
		{
			ULTALPR_SDK_PRINT_INFO("Stream #%d Standard exception caught: %s", index, e.what());
		}
	}
	ULTALPR_SDK_PRINT_INFO("Stream #%d Process for this stream has ended", index);
}


static void latest_frames(const std::string rtsp_path, bool usePipeline, std::string encoding, std::atomic<float> &fs, std::atomic<int> &frame_width, std::atomic<int> &frame_height, std::shared_ptr<FrameQueue> frame_queue_ptr, std::shared_ptr<cv::VideoCapture> cap_ptr, std::atomic<bool> &end_thread, std::atomic<double> &start_time, std::atomic<bool> &stream_started, std::atomic<int64_t> &last_frame_ms, int index)
{
	start_time.store(cv::getTickCount(), std::memory_order_release);
	stream_started.store(false, std::memory_order_release);

	if (usePipeline)
	{
#if defined(JETSON) || defined(ORIN)
		std::string pipeline = "rtspsrc location=\"" + rtsp_path + "\" ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=True max-buffers=1 ";
		if (encoding == "H264")
		{
			pipeline = "rtspsrc location=\"" + rtsp_path + "\" ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=True max-buffers=1 ";
		}
#else
		std::string pipeline = "uridecodebin uri=\"" + rtsp_path + "\" ! videoconvert ! appsink";
#endif
		ULTALPR_SDK_PRINT_INFO("Stream #%d Pipeline : %s", index, pipeline.c_str());
		stream_started.store(cap_ptr->open(pipeline, CAP_GSTREAMER), std::memory_order_release);
	}
	else
	{
#ifdef RPI
		stream_started.store(cap_ptr->open(rtsp_path, CAP_FFMPEG), std::memory_order_release);
#else
		stream_started.store(cap_ptr->open(rtsp_path, CAP_FFMPEG, {CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY}), std::memory_order_release);
#endif
	}

	if (stream_started.load(std::memory_order_acquire))
	{
		configure_capture(cap_ptr, index);
		ULTALPR_SDK_PRINT_INFO("Stream #%d RTSP Stream initialised", index);
		fs.store(cap_ptr->get(CAP_PROP_FPS), std::memory_order_release);
		ULTALPR_SDK_PRINT_INFO("Stream #%d RTSP Stream FPS %.2f", index, fs.load(std::memory_order_acquire));
		frame_width.store(cap_ptr->get(CAP_PROP_FRAME_WIDTH), std::memory_order_release);
		frame_height.store(cap_ptr->get(CAP_PROP_FRAME_HEIGHT), std::memory_order_release);
		ULTALPR_SDK_PRINT_INFO("Stream #%d RTSP Stream Resolution %dx%d", index, frame_width.load(std::memory_order_acquire), frame_height.load(std::memory_order_acquire));
		last_frame_ms.store(nowMs(), std::memory_order_release);
	}
	else
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d Failed to open RTSP stream", index);
		end_thread.store(true, std::memory_order_release);
	}
	while (!end_thread.load(std::memory_order_acquire))
	{
		start_time.store(cv::getTickCount(), std::memory_order_release);
		if (!cap_ptr->isOpened())
		{
			ULTALPR_SDK_PRINT_INFO("Stream #%d Stream not open anymore", index);
			end_thread.store(true, std::memory_order_release);
			break;
		}
		else
		{
			auto new_frame = std::make_shared<cv::Mat>();
			if (!cap_ptr->read(*new_frame))
			{
				ULTALPR_SDK_PRINT_INFO("Stream #%d Failed to read frame", index);
				end_thread.store(true, std::memory_order_release);
				break;
			}
			else
			{
				frame_mtx[index].lock();
				frame_queue_ptr->push(new_frame);
				frame_mtx[index].unlock();
				last_frame_ms.store(nowMs(), std::memory_order_release);
			}
		}
	}
	cap_ptr->release();
	ULTALPR_SDK_PRINT_INFO("Stream #%d Frames thread ended", index);
}

static void process_image_result(UltAlprSdkResult result, Mat frame, int index, const std::string server_url, int port, int relay, std::string logtype, std::string description, std::string ftp, float minimumConfidence, bool determineDirection, bool trackingEnabled, std::vector<Vehicle> &vehicles, bool isShowFrame, bool pushRawData)
{
	Mat original_frame = frame.clone();
	int width = frame.size().width;
	int height = frame.size().height;
	bool send_frame = false;
	std::vector<std::string> messages;
	const std::string &json_ = result.json();
	if (!json_.empty())
	{
		std::vector<Vehicle> new_vehicle_detections;
		Document document;
		document.Parse(json_.c_str());
		time_t curr_time_log;
		tm *curr_tm_log;
		char date_time_string_log[100];
		time(&curr_time_log);
		curr_tm_log = localtime(&curr_time_log);
		strftime(date_time_string_log, 50, "%Y-%m-%d %H:%M:%S", curr_tm_log);
		if (document.HasMember("plates"))
		{
			// ULTALPR_SDK_PRINT_INFO("time: %s, result: %s", date_time_string_log, json_.c_str());
			const Value &plates = document["plates"];
			for (SizeType i = 0; i < plates.Size(); i++)
			{
				cv::Rect carRect;
				float car_confidence = 30;
				std::string body_style = "car";
				float body_style_confidence = 0;
				if (plates[i].HasMember("car"))
				{

					car_confidence = plates[i]["car"]["confidence"].GetFloat();
					if (car_confidence > 50)
					{
						const Value &box = plates[i]["car"]["warpedBox"];
						int thickness = 2;
						Point p1((int)box[0].GetFloat(), (int)box[1].GetFloat());
						Point p2((int)box[2].GetFloat(), (int)box[3].GetFloat());
						Point p3((int)box[4].GetFloat(), (int)box[5].GetFloat());
						Point p4((int)box[6].GetFloat(), (int)box[7].GetFloat());
						carRect = Rect(
							(int)box[0].GetFloat(),
							(int)box[1].GetFloat(),
							std::max(
								std::abs((int)(box[4].GetFloat() - box[0].GetFloat())),
								std::abs((int)(box[6].GetFloat() - box[2].GetFloat()))),
							std::max(
								std::abs((int)(box[5].GetFloat() - box[1].GetFloat())),
								std::abs((int)(box[7].GetFloat() - box[3].GetFloat()))));
					}
					if (plates[i]["car"].HasMember("bodyStyle"))
					{
						body_style = plates[i]["car"]["bodyStyle"][0]["name"].GetString();
						body_style_confidence = plates[i]["car"]["bodyStyle"][0]["confidence"].GetFloat();
					}
				}

				if (plates[i].HasMember("text"))
				{

					std::string license_number = plates[i]["text"].GetString();
					float plateOcrConfidence = plates[i]["confidences"][0].GetFloat();
					float plateDetectionConfidence = plates[i]["confidences"][1].GetFloat();
					float min_confidence = 100.0;
					int min_index = -1;
					for (int j = 0; j < license_number.length(); j++)
					{
						if (plates[i]["confidences"][j + 2].GetFloat() < min_confidence)
						{
							min_confidence = plates[i]["confidences"][j + 2].GetFloat();
							min_index = j;
						}
					}
					int thickness = 2;
					const Value &box = plates[i]["warpedBox"];
					Point p1((int)box[0].GetFloat(), (int)box[1].GetFloat());
					Point p2((int)box[2].GetFloat(), (int)box[3].GetFloat());
					Point p3((int)box[4].GetFloat(), (int)box[5].GetFloat());
					Point p4((int)box[6].GetFloat(), (int)box[7].GetFloat());
					
					Vehicle vehicle_detection;
					vehicle_detection.vehicle_number = license_number;
					if (std::regex_match(license_number, std::regex("^(AN|AP|AR|AS|BR|BL|CH|CG|DD|DL|GA|GJ|HR|HP|JK|JH|KA|KL|LA|LD|MP|MH|MN|ML|MZ|NL|OD|PY|PB|RJ|SK|TN|TS|TR|UP|UK|WB)[0-9]{1,2}[A-Z0]{0,3}[0-9]{4}$|^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")))
					{
						vehicle_detection.valid_vehicle_number = license_number;
						vehicle_detection.valid_confidence = plateOcrConfidence;
					}
					else
					{
						vehicle_detection.valid_vehicle_number = "";
					}
					vehicle_detection.confidence = plateOcrConfidence;
					vehicle_detection.bounding_box = Rect(
						(int)box[0].GetFloat(),
						(int)box[1].GetFloat(),
						std::max(
							std::abs((int)(box[4].GetFloat() - box[0].GetFloat())),
							std::abs((int)(box[6].GetFloat() - box[2].GetFloat()))),
						std::max(
							std::abs((int)(box[5].GetFloat() - box[1].GetFloat())),
							std::abs((int)(box[7].GetFloat() - box[3].GetFloat()))));
					vehicle_detection.centroid = cv::Point((vehicle_detection.bounding_box.x + vehicle_detection.bounding_box.width / 2), (vehicle_detection.bounding_box.y + vehicle_detection.bounding_box.height / 2));
					vehicle_detection.vehicle_bounding_box = carRect;
					vehicle_detection.vehicle_centroid = cv::Point((vehicle_detection.vehicle_bounding_box.x + vehicle_detection.vehicle_bounding_box.width / 2), (vehicle_detection.vehicle_bounding_box.y + vehicle_detection.vehicle_bounding_box.height / 2));
					vehicle_detection.published = false;
					vehicle_detection.captured_frame = original_frame.clone();
					vehicle_detection.passed_through_centre = false;
					vehicle_detection.hidden_number_plate = false;
					vehicle_detection.body_style = body_style;
					vehicle_detection.body_style_confidence = body_style_confidence;
					vehicle_detection.message_id = print_random_string(10);
					if (vehicle_detection.vehicle_number.length() >= 8)
					{
						new_vehicle_detections.push_back(vehicle_detection);
					}

					if (license_number.size() >= 8)
					{
						if (std::regex_match(license_number, std::regex("^(AN|AP|AR|AS|BR|BL|CH|CG|DD|DL|GA|GJ|HR|HP|JK|JH|KA|KL|LA|LD|MP|MH|MN|ML|MZ|NL|OD|PY|PB|RJ|SK|TN|TS|TR|UP|UK|WB)[0-9]{1,2}[A-Z0]{0,3}[0-9]{4}$|^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")))
						{
							if (plateOcrConfidence > minimumConfidence)
							{
								if (min_confidence > minimumConfidence)
								{
									ULTALPR_SDK_PRINT_INFO("Stream #%d time: %s, result: %s, confidence %f, %s", index, date_time_string_log, license_number.c_str(), plateOcrConfidence, json_.c_str());
								}
								else
								{
									ULTALPR_SDK_PRINT_INFO("Stream #%d time: %s, result: %s, confidence %f, %s", index, date_time_string_log, license_number.c_str(), min_confidence, "Low character confidence");
								}
							}
							else
							{
								ULTALPR_SDK_PRINT_INFO("Stream #%d time: %s, result: %s, confidence %f, %s", index, date_time_string_log, license_number.c_str(), plateOcrConfidence, "Low confidence");
							}
						}
						else
						{
							ULTALPR_SDK_PRINT_INFO("Stream #%d time: %s, result: %s, %s", index, date_time_string_log, license_number.c_str(), "Invalid vehicle number");
						}
					}
					else
					{
						// ULTALPR_SDK_PRINT_INFO("Stream #%d time: %s, result: %s, %s", index, date_time_string_log, license_number.c_str(), "Vehicle number is too short" );
					}
				}
				
			}
			// ULTALPR_SDK_PRINT_INFO("result: %s", json_.c_str());
		}
		else
		{
			// ULTALPR_SDK_PRINT_INFO("result: %s", json_.c_str());
		}

		std::chrono::time_point<std::chrono::system_clock> compare_time = std::chrono::system_clock::now();

		Rect roi = Rect(
			(int)width * 0.2,
			(int)height * 0.2,
			(int)width * 0.6,
			(int)height * 0.6);
		if (pushRawData)
		{
			roi = Rect(
				(int)width * 0.1,
				(int)height * 0.1,
				(int)width * 0.8,
				(int)height * 0.8);
		}

		for (int j = 0; j < new_vehicle_detections.size(); j++)
		{
			bool isNewDetection = true;
			for (int i = 0; i < vehicles.size(); i++)
			{
				int levenshtein_distance = levenshteinDistance(new_vehicle_detections[j].vehicle_number, vehicles[i].vehicle_number);
				cv::Rect intersection = new_vehicle_detections[j].bounding_box & vehicles[i].bounding_box;
				cv::Rect unionRect = new_vehicle_detections[j].bounding_box | vehicles[i].bounding_box;
				float iou = (float)intersection.area() / unionRect.area();
				int centroid_threshold = std::min(vehicles[i].bounding_box.width, vehicles[i].bounding_box.height);
				int vehicle_centroid_threshold = height / 8;
				int adjusted_threshold = height / 5;

				if (
					(
						(
							iou > 0.5 ||
							cv::norm(new_vehicle_detections[j].centroid - vehicles[i].centroid) < centroid_threshold) &&
						!new_vehicle_detections[j].hidden_number_plate &&
						!vehicles[i].hidden_number_plate) ||
					(levenshtein_distance < 3 &&
					 new_vehicle_detections[j].vehicle_number != "" &&
					 vehicles[i].vehicle_number != "" &&
					 cv::norm(new_vehicle_detections[j].centroid - vehicles[i].centroid) < adjusted_threshold)
					
				)
				{
					isNewDetection = false;
					if (!new_vehicle_detections[j].hidden_number_plate)
					{
						// Update vehicle number with higher confidence one
						if (new_vehicle_detections[j].confidence > vehicles[i].confidence)
						{
							vehicles[i].vehicle_number = new_vehicle_detections[j].vehicle_number;
							vehicles[i].confidence = new_vehicle_detections[j].confidence;
							if (new_vehicle_detections[j].valid_vehicle_number != "")
							{
								vehicles[i].valid_vehicle_number = new_vehicle_detections[j].valid_vehicle_number;
							}
							vehicles[i].captured_frame = new_vehicle_detections[j].captured_frame;
						}
						if (new_vehicle_detections[j].valid_confidence > vehicles[i].valid_confidence)
						{
							vehicles[i].valid_confidence = new_vehicle_detections[j].valid_confidence;
							vehicles[i].valid_vehicle_number = new_vehicle_detections[j].valid_vehicle_number;
							vehicles[i].valid_frame = new_vehicle_detections[j].captured_frame;
						}
						if (new_vehicle_detections[j].body_style_confidence > vehicles[i].body_style_confidence)
						{
							vehicles[i].body_style_confidence = new_vehicle_detections[j].body_style_confidence;
							vehicles[i].body_style = new_vehicle_detections[j].body_style;
						}
						vehicles[i].bounding_box = new_vehicle_detections[j].bounding_box;
						vehicles[i].centroid = new_vehicle_detections[j].centroid;
						if (!vehicles[i].passed_through_centre)
						{
							if (roi.contains(vehicles[i].centroid))
							{
								vehicles[i].passed_through_centre = true;
								vehicles[i].centre_frame = new_vehicle_detections[j].captured_frame;
							}
						}
						if (new_vehicle_detections[j].valid_vehicle_number != "")
						{
							vehicles[i].valid_centre_frame = new_vehicle_detections[j].captured_frame;
						}
						vehicles[i].final_position = cv::Point(new_vehicle_detections[j].centroid.x, new_vehicle_detections[j].centroid.y);
					}
					if (!new_vehicle_detections[j].vehicle_bounding_box.empty())
					{
						vehicles[i].vehicle_bounding_box = new_vehicle_detections[j].vehicle_bounding_box;
						vehicles[i].vehicle_centroid = new_vehicle_detections[j].vehicle_centroid;
					}
					vehicles[i].tracking_count++;
					std::chrono::time_point<std::chrono::system_clock> current_time = std::chrono::system_clock::now();
					vehicles[i].last_seen_at = current_time;
					if (!vehicles[i].published)
					{
						vehicles[i].published_at = current_time;
					}
					break;
				}
			}

			if (isNewDetection)
			{
				new_vehicle_detections[j].tracking_count = 1;
				std::chrono::time_point<std::chrono::system_clock> current_time = std::chrono::system_clock::now();
				new_vehicle_detections[j].first_seen_at = current_time;
				new_vehicle_detections[j].last_seen_at = current_time;
				new_vehicle_detections[j].initial_position = cv::Point(new_vehicle_detections[j].centroid.x, new_vehicle_detections[j].centroid.y);
				vehicles.push_back(new_vehicle_detections[j]);
			}
		}

		std::vector<int> indices_to_delete;
		for (int i = 0; i < vehicles.size(); i++)
		{
			// Increase this if boom barrier may come in between
			if (std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].last_seen_at).count() > 3000)
			{
				indices_to_delete.push_back(i);
			}
			else
			{
				if (std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].published_at).count() > 500 || (!vehicles[i].published && std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].first_seen_at).count() > 1000) ||
					(pushRawData && (std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].published_at).count() > 300 || (!vehicles[i].published && std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].first_seen_at).count() > 300))))
				{
					if (vehicles[i].passed_through_centre)
					{
						std::time_t first_seen_at_t = std::chrono::system_clock::to_time_t(vehicles[i].first_seen_at);
						std::time_t last_seen_at_t = std::chrono::system_clock::to_time_t(vehicles[i].last_seen_at);
						char first_seen_at_string[100];
						strftime(first_seen_at_string, 50, "%Y-%m-%d %H:%M:%S", localtime(&first_seen_at_t));
						char last_seen_at_string[100];
						strftime(last_seen_at_string, 50, "%Y-%m-%d %H:%M:%S", localtime(&last_seen_at_t));
						char save_timestamp_string[100];
						strftime(save_timestamp_string, 50, "%Y%m%d%H%M%S", localtime(&first_seen_at_t));
						ULTALPR_SDK_PRINT_INFO("Stream #%d time: %s, result: %s, vehicle_number: %s, confidence %f, body_style %s, body_style_confidence %f, first_seen_at %s, last_seen_at %s, coordinates %d,%d,%d,%d, initial %d,%d, final %d,%d", index, date_time_string_log, vehicles[i].valid_vehicle_number.c_str(), vehicles[i].vehicle_number.c_str(), vehicles[i].confidence, vehicles[i].body_style.c_str(), vehicles[i].body_style_confidence, first_seen_at_string, last_seen_at_string, vehicles[i].bounding_box.x, vehicles[i].bounding_box.y, vehicles[i].bounding_box.x + vehicles[i].bounding_box.width, vehicles[i].bounding_box.y + vehicles[i].bounding_box.height, vehicles[i].initial_position.x, vehicles[i].initial_position.y, vehicles[i].final_position.x, vehicles[i].final_position.y);
						char saved_path[200];
						strcpy(saved_path, "/home/parksmart/anpr/images/");
						strcat(saved_path, save_timestamp_string);
						strcat(saved_path, "-");
						strcat(saved_path, vehicles[i].vehicle_number.c_str());
						strcat(saved_path, ".jpg");
						std::string saved_path_string(saved_path);
						if (ftp != "")
						{
							std::string http_url = "http:/";
							http_url += "/" + ftp + ":8196/";
							saved_path_string.replace(0, 28, http_url);
						}
						bool writeSuccessAnpr;
						struct stat buffer;
						if (stat(saved_path_string.c_str(), &buffer) == 0)
						{
							writeSuccessAnpr = true;
						}
						else
						{
							if (!vehicles[i].valid_centre_frame.empty())
							{
								writeSuccessAnpr = imwrite(saved_path, vehicles[i].valid_centre_frame);
							}
							else
							{
								writeSuccessAnpr = imwrite(saved_path, vehicles[i].centre_frame);
							}
						}
						if (writeSuccessAnpr)
						{
							std::string socket_message = "{";
							if (vehicles[i].valid_vehicle_number != "")
							{
								socket_message += "\"vehicle_number\":\"";
								socket_message += vehicles[i].valid_vehicle_number;
								socket_message += "\",";
							}
							else
							{
								socket_message += "\"vehicle_number\":\"";
								socket_message += vehicles[i].vehicle_number;
								socket_message += "\",";
							}
							socket_message += "\"image_location\":\"";
							socket_message += saved_path_string;
							socket_message += "\",";
							
							socket_message += "\"add_to_surveillance\":";
							socket_message += std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].last_seen_at).count() > 1000 && !pushRawData);
							socket_message += ",";

							socket_message += "\"relay\":";
							socket_message += std::to_string(relay);
							socket_message += ",";

							socket_message += "\"lane_id\":";
							socket_message += std::to_string(index);
							socket_message += ",";

							socket_message += "\"message_id\":\"";
							socket_message += vehicles[i].message_id;
							socket_message += "\",";

							socket_message += "\"confidence\":";
							socket_message += std::to_string(vehicles[i].confidence);
							socket_message += ",";

							
							socket_message += "\"raw_image_location\":\"";
							socket_message += saved_path_string;
							socket_message += "\",";

							socket_message += "\"bbox\":\"";
							socket_message += std::to_string(vehicles[i].bounding_box.x) + "," + std::to_string(vehicles[i].bounding_box.y) + ",";
							socket_message += std::to_string(vehicles[i].bounding_box.x + vehicles[i].bounding_box.width) + "," + std::to_string(vehicles[i].bounding_box.y) + ",";
							socket_message += std::to_string(vehicles[i].bounding_box.x + vehicles[i].bounding_box.width) + "," + std::to_string(vehicles[i].bounding_box.y + vehicles[i].bounding_box.height) + ",";
							socket_message += std::to_string(vehicles[i].bounding_box.x) + "," + std::to_string(vehicles[i].bounding_box.y + vehicles[i].bounding_box.height);
							socket_message += "\",";

							socket_message += "\"direction\":\"";
							if (determineDirection)
							{
								if (vehicles[i].final_position.y - vehicles[i].initial_position.y > height * 0.1)
								{
									socket_message += logtype;
								}
								else
								{
									if (vehicles[i].initial_position.y - vehicles[i].final_position.y > height * 0.1)
									{
										if (logtype == "ENTRY")
										{
											socket_message += "EXIT";
										}
										else
										{
											socket_message += "ENTRY";
										}
									}
									else
									{
										socket_message += logtype;
									}
								}
							}
							else
							{
								socket_message += logtype;
							}
							socket_message += "\",";

							socket_message += "\"description\":\"";
							socket_message += description;
							socket_message += "\"}";

							if (vehicles[i].body_style.compare("motorcycle") != 0)
							{
								messages.push_back(socket_message);
							}
							send_frame = true;
							std::chrono::time_point<std::chrono::system_clock> current_time = std::chrono::system_clock::now();
							vehicles[i].published_at = current_time;
							vehicles[i].published = true;
						}
						else
						{
							ULTALPR_SDK_PRINT_INFO("Stream #%d Write failed %s", index, saved_path);
						}
					}
				}
			}
		}

		while (indices_to_delete.size() > 0)
		{
			int index_to_delete = indices_to_delete.back();
			vehicles.erase(vehicles.begin() + index_to_delete);
			indices_to_delete.pop_back();
		}

		if (isShowFrame)
		{
			cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
			for (int i = 0; i < vehicles.size(); i++)
			{
				if (vehicles[i].tracking_count > 5 && std::chrono::duration_cast<std::chrono::milliseconds>(compare_time - vehicles[i].last_seen_at).count() < 1000)
				{
					cv::rectangle(frame, vehicles[i].bounding_box, cv::Scalar(255, 0, 0), 2);
					cv::putText(frame, vehicles[i].vehicle_number, cv::Point(vehicles[i].bounding_box.x, vehicles[i].bounding_box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
					cv::putText(frame, "Confidence: " + std::to_string(vehicles[i].confidence), cv::Point(vehicles[i].bounding_box.x, vehicles[i].bounding_box.y - 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
					if (!vehicles[i].vehicle_bounding_box.empty())
					{
						cv::rectangle(frame, vehicles[i].vehicle_bounding_box, cv::Scalar(255, 0, 0), 2);
					}
				}
			}
		}
	}

	if (send_frame)
	{
		for (int i = 0; i < messages.size(); i++)
		{
			auto message_thread = std::thread(send_message, server_url, port, messages[i], index);
			message_thread.detach();
		}
	}
}

#ifdef RPI
static void process_image(Mat frame, bool isParallelDeliveryEnabled, int index, const std::string server_url, int port, int relay, float &f, float *FPS, int &frame_count, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, float minimumConfidence, bool determineDirection, bool trackingEnabled, std::vector<Vehicle> &vehicles, int* fps_counter, bool isGpioInitialised, int gpio_pin, int default_gpio_level)
#else
static void process_image(Mat frame, bool isParallelDeliveryEnabled, int index, const std::string server_url, int port, int relay, float &f, float *FPS, int &frame_count, std::string logtype, std::string description, bool isShowFrame, bool pushRawData, std::string ftp, float minimumConfidence, bool determineDirection, bool trackingEnabled, std::vector<Vehicle> &vehicles, int* fps_counter)
#endif
{
	std::chrono::system_clock::time_point Tbegin, Tend;

	Tbegin = std::chrono::system_clock::now();

	bool processImage = true;
#ifdef RPI
	if (isGpioInitialised)
	{
		int vehicle_present = gpioRead(gpio_pin);
		// ULTALPR_SDK_PRINT_INFO("Stream #%d Vehicle on loop: %d", index, vehicle_present);
		if (vehicle_present == default_gpio_level)
		{
			processImage = false;
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
	}
#endif

	if (processImage)
	{
		UltAlprSdkResult result;
		ULTALPR_SDK_ASSERT((result = UltAlprSdkEngine::process(
								ULTALPR_SDK_IMAGE_TYPE_RGB24,
								frame.ptr(),
								frame.size().width,
								frame.size().height))
							   .isOK());

		const std::string &json_ = result.json();
		// ULTALPR_SDK_PRINT_INFO("result: %s", json_.c_str());
		if (isParallelDeliveryEnabled)
		{
			Document document;
			document.Parse(json_.c_str());
			if (document.HasMember("frame_id"))
			{
				int frame_id = document["frame_id"].GetInt();
				frame_id = frame_id % 100;
				frames[index][frame_id] = frame.clone();
				frame_id_index_map[frame_id] = index;
			}
		}

		// Print latest result
		if (!isParallelDeliveryEnabled && result.json())
		{ // for parallel delivery the result will be printed by the callback function
			process_image_result(result, frame, index, server_url, port, relay, logtype, description, ftp, minimumConfidence, determineDirection, trackingEnabled, vehicles, isShowFrame, pushRawData);
		}
	}

	Tend = std::chrono::system_clock::now();

	f = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tbegin).count();
	int i;
	if (f > 0.0)
		FPS[((frame_count++) & 0x0F)] = 1000.0 / f;
	for (f = 0.0, i = 0; i < 16; i++)
	{
		f += FPS[i];
	}
	putText(frame, cv::format("FPS %0.2f", f / 16), Point(5, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

	#if defined(JETSON) || defined(RPI)
	const int fps_threshold = 25;
	#else
	const int fps_threshold = 25;
	#endif

	(*fps_counter)++;
	if (*fps_counter >= fps_threshold) {
		ULTALPR_SDK_PRINT_INFO("Stream #%d Running at FPS %0.2f", index, f / 16);
		*fps_counter = 0;
	}

	
	Mat resized_down;
	resize(frame, resized_down, Size(960, 540));
	std::string window_name = "anpr";
	window_name += std::to_string(index);
	if (isShowFrame)
	{
		imshow(window_name, resized_down);
	}
	waitKey(1); // waits to display frame
}



static int send_message(const std::string server, const int port, const std::string message, int index)
{
	int sock = 0, valread, client_fd;
	struct sockaddr_in serv_addr;
	char buffer[1024] = {0};
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d: %s", index, "Socket creation error");
		return -1;
	}
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	// Convert IPv4 and IPv6 addresses from text to binary
	// form
	if (inet_pton(AF_INET, server.c_str(), &serv_addr.sin_addr) <= 0)
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d: %s", index, "Invalid address. Address not supported");
		return -1;
	}
	if ((client_fd = connect(sock, (struct sockaddr *)&serv_addr,
							 sizeof(serv_addr))) < 0)
	{
		ULTALPR_SDK_PRINT_INFO("Stream #%d: %s", index, "Connection Failed");
		return -1;
	}
	ULTALPR_SDK_PRINT_INFO("Stream #%d: Message %s", index, message.c_str());
	send(sock, message.c_str(), strlen(message.c_str()), 0);
	// closing the connected socket
	close(client_fd);
	return 0;
}

static int levenshteinDistance(const std::string &s1, const std::string &s2)
{
	// Create a matrix with rows representing s1 and columns representing s2
	std::vector<std::vector<int>> d(s1.size() + 1, std::vector<int>(s2.size() + 1));

	// Initialize the first row and column of the matrix
	for (int i = 0; i <= s1.size(); i++)
		d[i][0] = i;
	for (int j = 0; j <= s2.size(); j++)
		d[0][j] = j;

	// Fill in the rest of the matrix using the Levenshtein distance formula
	for (int i = 1; i <= s1.size(); i++)
	{
		for (int j = 1; j <= s2.size(); j++)
		{
			int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
			d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
		}
	}

	// The distance is the value in the bottom-right corner of the matrix
	return d[s1.size()][s2.size()];
}

const int MAX = 26;

static std::string print_random_string(int n)
{
	char alphabet[MAX] = {'a', 'b', 'c', 'd', 'e', 'f', 'g',
						  'h', 'i', 'j', 'k', 'l', 'm', 'n',
						  'o', 'p', 'q', 'r', 's', 't', 'u',
						  'v', 'w', 'x', 'y', 'z'};
	std::string res = "";
	for (int i = 0; i < n; i++)
	{
		res = res + alphabet[rand() % MAX];
	}
	return res;
}

static void printUsage(const std::string &message /*= ""*/)
{
	if (!message.empty())
	{
		ULTALPR_SDK_PRINT_ERROR("%s", message.c_str());
	}

	ULTALPR_SDK_PRINT_INFO(
		"\n********************************************************************************\n"
		"recognizer\n"
		"\t--image <path-to-image-with-to-recognize> \n"
		"\t[--assets <path-to-assets-folder>] \n"
		"\t[--charset <recognition-charset:latin/korean/chinese>] \n"
		"\t[--car_noplate_detect_enabled <whether-to-enable-detecting-cars-with-no-plate:true/false>] \n"
		"\t[--ienv_enabled <whether-to-enable-IENV:true/false>] \n"
		"\t[--openvino_enabled <whether-to-enable-OpenVINO:true/false>] \n"
		"\t[--openvino_device <openvino_device-to-use>] \n"
		"\t[--klass_lpci_enabled <whether-to-enable-LPCI:true/false>] \n"
		"\t[--klass_vcr_enabled <whether-to-enable-VCR:true/false>] \n"
		"\t[--klass_vmmr_enabled <whether-to-enable-VMMR:true/false>] \n"
		"\t[--klass_vbsr_enabled <whether-to-enable-VBSR:true/false>] \n"
		"\t[--parallel <whether-to-enable-parallel-mode:true / false>] \n"
		"\t[--rectify <whether-to-enable-rectification-layer:true / false>] \n"
		"\t[--tokenfile <path-to-license-token-file>] \n"
		"\t[--tokendata <base64-license-token-data>] \n"
		"\n"
		"Options surrounded with [] are optional.\n"
		"\n"
		"--image: Path to the image(JPEG/PNG/BMP) to process. You can use default image at ../../../assets/images/lic_us_1280x720.jpg.\n\n"
		"--assets: Path to the assets folder containing the configuration files and models. Default value is the current folder.\n\n"
		"--charset: Defines the recognition charset (a.k.a alphabet) value (latin, korean, chinese...). Default: latin.\n\n"
		"--charset: Defines the recognition charset value (latin, korean, chinese...). Default: latin.\n\n"
		"--car_noplate_detect_enabled: Whether to detect and return cars with no plate. Default: false.\n\n"
		"--ienv_enabled: Whether to enable Image Enhancement for Night-Vision (IENV). More info about IENV at https://www.doubango.org/SDKs/anpr/docs/Features.html#image-enhancement-for-night-vision-ienv. Default: true for x86-64 and false for ARM.\n\n"
		"--openvino_enabled: Whether to enable OpenVINO. Tensorflow will be used when OpenVINO is disabled. Default: true.\n\n"
		"--openvino_device: Defines the OpenVINO device to use (CPU, GPU, FPGA...). More info at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html#openvino_device. Default: CPU.\n\n"
		"--npu_enabled: Whether to enable NPU acceleration (Amlogic, NXP...). Default: true.\n\n"
		"--trt_enabled: Whether to enable NVIDIA TensorRT acceleration. This will disable OpenVINO More info at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html#trt-enabled. Default: false.\n\n"
		"--klass_lpci_enabled: Whether to enable License Plate Country Identification (LPCI). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#license-plate-country-identification-lpci. Default: false.\n\n"
		"--klass_vcr_enabled: Whether to enable Vehicle Color Recognition (VCR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-color-recognition-vcr. Default: false.\n\n"
		"--klass_vmmr_enabled: Whether to enable Vehicle Make Model Recognition (VMMR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-make-model-recognition-vmmr. Default: false.\n\n"
		"--klass_vbsr_enabled: Whether to enable Vehicle Body Style Recognition (VBSR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-make-model-recognition-vbsr. Default: false.\n\n"
		"--parallel: Whether to enabled the parallel mode.More info about the parallel mode at https://www.doubango.org/SDKs/anpr/docs/Parallel_versus_sequential_processing.html. Default: true.\n\n"
		"--rectify: Whether to enable the rectification layer. More info about the rectification layer at https ://www.doubango.org/SDKs/anpr/docs/Rectification_layer.html. Default: true.\n\n"
		"--tokenfile: Path to the file containing the base64 license token if you have one. If not provided then, the application will act like a trial version. Default: null.\n\n"
		"--tokendata: Base64 license token if you have one. If not provided then, the application will act like a trial version. Default: null.\n\n"
		"********************************************************************************\n");
}

static bool parseArgs(int argc, char *argv[], std::map<std::string, std::string> &values)
{
	ULTALPR_SDK_ASSERT(argc > 0 && argv != nullptr);

	values.clear();
	values["--rtsp"] = "";
	values["--server"] = "";
	values["--logtype"] = "";
	values["--description"] = "";

	// Make sure the number of arguments is even
	if ((argc - 1) & 1)
	{
		ULTALPR_SDK_PRINT_ERROR("Number of args must be even");
		return false;
	}

	// Parsing
	for (int index = 1; index < argc; index += 2)
	{
		std::string key = argv[index];
		if (key.size() < 2 || key[0] != '-' || key[1] != '-')
		{
			ULTALPR_SDK_PRINT_ERROR("Invalid key: %s", key.c_str());
			return false;
		}
		if (key != "--rtsp")
		{
			if (key != "--server")
			{
				if (key != "--port")
				{
					if (key != "--relay")
					{
						if (key != "--logtype")
						{
							if (key != "--description")
							{
								values[key] = argv[index + 1];
							}
							else
							{
								if (values["--description"] != "")
								{
									values["--description"] += ",";
								}
								values["--description"] += (argv[index + 1]);
							}
						}
						else
						{
							if (values["--logtype"] != "")
							{
								values["--logtype"] += ",";
							}
							values["--logtype"] += (argv[index + 1]);
						}
					}
					else
					{
						if (values["--relay"] != "")
						{
							values["--relay"] += ",";
						}
						values["--relay"] += (argv[index + 1]);
					}
				}
				else
				{
					if (values["--port"] != "")
					{
						values["--port"] += ",";
					}
					values["--port"] += (argv[index + 1]);
				}
			}
			else
			{
				if (values["--server"] != "")
				{
					values["--server"] += ",";
				}
				values["--server"] += (argv[index + 1]);
			};
		}
		else
		{
			if (values["--rtsp"] != "")
			{
				values["--rtsp"] += ",";
			}
			values["--rtsp"] += (argv[index + 1]);
		}
	}

	return true;
}
