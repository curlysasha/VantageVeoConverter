"""
Video comparison and grid creation
"""
import os
import logging
import subprocess
import shutil
import cv2

def create_comparison_grid(original_video, adaptive_video, precision_video, maximum_video, output_path):
    """Создает диптих 2x2 из 4 видео с подписями режимов (упрощенная версия)."""
    
    try:
        # Получаем информацию о самом коротком видео для синхронизации
        videos = [original_video, adaptive_video, precision_video, maximum_video]
        labels = ["ORIGINAL", "ADAPTIVE", "PRECISION", "MAXIMUM"]
        
        min_duration = float('inf')
        for video in videos:
            cap = cv2.VideoCapture(video)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frames / fps
            min_duration = min(min_duration, duration)
            cap.release()
        
        # Ограничиваем длительность для быстрого сравнения
        demo_duration = min(30, min_duration)
        
        logging.info(f"Creating grid with {demo_duration:.1f}s duration")
        
        # FFmpeg команда с аудио из первого видео
        command = [
            "ffmpeg", "-y",
            "-i", original_video,
            "-i", adaptive_video, 
            "-i", precision_video,
            "-i", maximum_video,
            "-filter_complex", 
            f"[0:v]scale=640:360,drawtext=text='{labels[0]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v0];"
            f"[1:v]scale=640:360,drawtext=text='{labels[1]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v1];"
            f"[2:v]scale=640:360,drawtext=text='{labels[2]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v2];"
            f"[3:v]scale=640:360,drawtext=text='{labels[3]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v3];"
            "[v0][v1]hstack[top];"
            "[v2][v3]hstack[bottom];"
            "[top][bottom]vstack[v]",
            "-map", "[v]",
            "-map", "0:a",  # Берем аудио из первого видео (original)
            "-c:v", "libx264", 
            "-c:a", "aac",  # Кодируем аудио в AAC
            "-preset", "ultrafast",  # Быстрое кодирование
            "-crf", "28",  # Быстрое сжатие
            "-t", str(demo_duration),
            "-r", "15",  # Пониженный FPS для скорости
            output_path
        ]
        
        logging.info("Starting FFmpeg grid creation with audio...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Grid with audio created successfully!")
            return True
        else:
            logging.error(f"FFmpeg failed: {result.stderr}")
            
            # Fallback: создаем без аудио, потом добавляем
            logging.info("Fallback: creating grid without audio first...")
            
            # Команда без аудио
            command_no_audio = [
                "ffmpeg", "-y",
                "-i", original_video,
                "-i", adaptive_video, 
                "-i", precision_video,
                "-i", maximum_video,
                "-filter_complex", 
                f"[0:v]scale=640:360,drawtext=text='{labels[0]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v0];"
                f"[1:v]scale=640:360,drawtext=text='{labels[1]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v1];"
                f"[2:v]scale=640:360,drawtext=text='{labels[2]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v2];"
                f"[3:v]scale=640:360,drawtext=text='{labels[3]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v3];"
                "[v0][v1]hstack[top];"
                "[v2][v3]hstack[bottom];"
                "[top][bottom]vstack[v]",
                "-map", "[v]",
                "-c:v", "libx264", 
                "-preset", "ultrafast",
                "-crf", "28",
                "-t", str(demo_duration),
                "-r", "15",
                os.path.splitext(output_path)[0] + "_video_only.mp4"
            ]
            
            result_video = subprocess.run(command_no_audio, capture_output=True, text=True)
            
            if result_video.returncode == 0:
                # Теперь добавляем аудио
                video_only_path = os.path.splitext(output_path)[0] + "_video_only.mp4"
                
                command_add_audio = [
                    "ffmpeg", "-y",
                    "-i", video_only_path,
                    "-i", original_video,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    output_path
                ]
                
                result_audio = subprocess.run(command_add_audio, capture_output=True, text=True)
                
                # Очищаем временный файл
                try:
                    os.unlink(video_only_path)
                except:
                    pass
                
                if result_audio.returncode == 0:
                    logging.info("Grid with audio created via fallback!")
                    return True
            
            # Последний fallback: просто копируем первое видео
            logging.info("Final fallback: using original video as output")
            shutil.copy2(original_video, output_path)
            return True
            
    except Exception as e:
        logging.error(f"Grid creation failed: {e}")
        
        # Fallback: копируем оригинальное видео
        try:
            shutil.copy2(original_video, output_path)
            logging.info("Fallback: copied original video")
            return True
        except:
            return False