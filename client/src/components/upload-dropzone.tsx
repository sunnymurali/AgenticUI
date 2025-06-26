import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { FileText, Upload, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface UploadDropzoneProps {
  agentId: string;
}

interface UploadFile {
  file: File;
  progress: number;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

export function UploadDropzone({ agentId }: UploadDropzoneProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: ({ file }: { file: File }) => api.uploadDocument(agentId, file),
    onSuccess: (result, { file }) => {
      setFiles(prev => prev.map(f => 
        f.file === file 
          ? { ...f, status: "success" as const, progress: 100 }
          : f
      ));
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      queryClient.invalidateQueries({ queryKey: ["agents", agentId] });
      toast({
        title: "Success",
        description: `Uploaded ${result.uploaded_chunks} chunks from ${file.name}`,
      });
    },
    onError: (error: Error, { file }) => {
      setFiles(prev => prev.map(f => 
        f.file === file 
          ? { ...f, status: "error" as const, error: error.message }
          : f
      ));
      toast({
        title: "Error",
        description: `Failed to upload ${file.name}: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const supportedTypes = [
      "application/pdf",
      "text/plain",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "image/png",
      "image/jpeg",
      "image/jpg"
    ];
    
    const newFiles = acceptedFiles
      .filter(file => supportedTypes.includes(file.type) || 
        file.name.endsWith('.txt') || file.name.endsWith('.pdf') || 
        file.name.endsWith('.doc') || file.name.endsWith('.docx') ||
        file.name.endsWith('.png') || file.name.endsWith('.jpg') || file.name.endsWith('.jpeg'))
      .map(file => ({
        file,
        progress: 0,
        status: "pending" as const,
      }));

    if (newFiles.length !== acceptedFiles.length) {
      toast({
        title: "Warning",
        description: "Only PDF, DOC, DOCX, TXT, PNG, and JPG files are supported",
        variant: "destructive",
      });
    }

    setFiles(prev => [...prev, ...newFiles]);

    // Start uploading
    newFiles.forEach(({ file }) => {
      setFiles(prev => prev.map(f => 
        f.file === file 
          ? { ...f, status: "uploading" as const, progress: 45 }
          : f
      ));
      uploadMutation.mutate({ file });
    });
  }, [uploadMutation, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'image/png': ['.png'],        // NEW
      'image/jpeg': ['.jpg', '.jpeg'] // NEW
    },
    multiple: true,
  });

  const removeFile = (fileToRemove: File) => {
    setFiles(prev => prev.filter(f => f.file !== fileToRemove));
  };

  return (
    <div className="space-y-4">
      <Card 
        {...getRootProps()} 
        className={`border-2 border-dashed p-8 text-center cursor-pointer transition-colors duration-200 ${
          isDragActive ? 'border-primary bg-blue-50' : 'border-gray-300 hover:border-primary'
        }`}
      >
        <input {...getInputProps()} />
        <div className="mx-auto w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
          <FileText className="text-2xl text-slate-400" />
        </div>
        <h4 className="text-lg font-medium text-slate-800 mb-2">Upload Documents</h4>
        <p className="text-slate-600 mb-4">
          {isDragActive 
            ? "Drop your files here..." 
            : "Drag and drop your files here, or click to browse"
          }
        </p>
        <p className="text-xs text-slate-500 mb-4">Supported formats: PDF, DOC, DOCX, TXT, PNG, JPG</p>
        <Button type="button" className="bg-primary text-white hover:bg-blue-700">
          <Upload className="mr-2" size={16} />
          Choose Files
        </Button>
      </Card>

      {files.length > 0 && (
        <Card className="p-4">
          <h4 className="font-medium text-slate-800 mb-4">Upload Progress</h4>
          <div className="space-y-3">
            {files.map(({ file, progress, status, error }, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <FileText className="text-red-600" size={16} />
                    <span className="text-sm font-medium text-slate-700">{file.name}</span>
                    {status === "success" && <span className="text-xs text-green-600">✓ Complete</span>}
                    {status === "error" && <span className="text-xs text-red-600">✗ Failed</span>}
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-slate-500">
                      {status === "success" ? "100%" : `${progress}%`}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(file)}
                      className="p-1"
                    >
                      <X size={14} />
                    </Button>
                  </div>
                </div>
                <Progress 
                  value={status === "success" ? 100 : progress} 
                  className="h-2"
                />
                {error && (
                  <p className="text-xs text-red-600">{error}</p>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
