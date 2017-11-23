def compute():
    with open('mobile_test/sent_data_1.txt') as data:
        for sentence in data:
            print(sentence) 
            break

def main():
    compute()


if __name__ == '__main__':
    main()

