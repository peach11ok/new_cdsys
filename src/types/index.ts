export interface User {
  id: string;
  username: string;
  role: 'root' | 'normal';
}

export interface DetectionRecord {
  id: string;
  userId: string;
  username?: string;
  timestamp: string;
  inputImages: {
    image1: string;
    image2: string;
  };
  models: {
    detectionModel: string;
    segmentationModel: string;
  };
  results: {
    changeDetectionImage: string;
    segmentationImages: {
      image1: string;
      image2: string;
    };
    changedAreas: string[];
    changeTypes: string[];
  };
}

export interface DetectionResult {
  changedAreas: string[];
  confidence: number;
  segmentationData: {
    image1: string[];
    image2: string[];
  };
  changeDetectionImage: string;
  segmentationImages: {
    image1: string;
    image2: string;
  };
}

export interface StoredUser extends User {
  password: string;
  avatar?: string;
  detectionCount: number;
  createdAt: string;
  updatedAt: string;
} 