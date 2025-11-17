export interface UserInfo {
  token: string;
  tokenExpiration: number;
  username: string;
  role: string;
}

export interface LoginParams {
  username: string;
  password: string;
}