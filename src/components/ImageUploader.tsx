import React, { useRef, useState } from 'react';
import { CloudUpload } from '@mui/icons-material';

interface Props {
  label: string;
  onChange: (file: File | null) => void;
}

const ImageUploader: React.FC<Props> = ({ label, onChange }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string>('');

  const validateImageDimensions = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(img.src);
        resolve(img.width === 224 && img.height === 224);
      };
      img.src = URL.createObjectURL(file);
    });
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isValidDimensions = await validateImageDimensions(file);
      if (!isValidDimensions) {
        alert('请上传 224x224 像素的图片');
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        setPreview('');
        onChange(null);
        return;
      }

      // 创建预览URL
      const previewUrl = URL.createObjectURL(file);
      setPreview(previewUrl);
      onChange(file);
    }
  };

  // 组件卸载时清理预览URL
  React.useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  return (
    <div className="flex flex-col items-center">
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        accept="image/*"
        onChange={handleFileChange}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        className={`w-[224px] h-[224px] border-2 border-dashed rounded-lg 
          ${preview ? 'border-blue-500' : 'border-gray-300 hover:border-blue-500'}
          flex flex-col items-center justify-center relative overflow-hidden`}
      >
        {preview ? (
          <>
            <img
              src={preview}
              alt="预览"
              className="absolute inset-0 w-[224px] h-[224px] object-none"
            />
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
              <span className="text-white text-sm">点击更换图片</span>
            </div>
          </>
        ) : (
          <>
            <CloudUpload className="h-8 w-8 text-gray-400" />
            <span className="mt-2 text-sm text-gray-500">{label}</span>
            <span className="mt-1 text-xs text-gray-400">仅支持 224x224 像素的图片</span>
          </>
        )}
      </button>
    </div>
  );
};

export default ImageUploader; 